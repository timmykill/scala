import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix,
  MatrixEntry, RowMatrix, BlockMatrix}
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.mllib.linalg.{DenseVector, Matrices, DenseMatrix,
  Matrix}
import org.nspl._
import org.nspl.awtrenderer._
import scala.util.Random
import scala.math.{pow, sqrt, max}
import java.awt.{Panel, Frame, Graphics}
import javax.imageio.ImageIO
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StructType, StructField, IntegerType,
  DoubleType, LongType}
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import scala.collection.immutable.NumericRange
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.SparkContext
import scala.collection.mutable.ArraySeq

package SimpleApp {


  object SimpleApp {
    def stamp() = {
      System.currentTimeMillis / 1000
    }

    def main(args: Array[String]): Unit = {
      val config : Map[String, Double] = upickle.default.read[Map[String, Double]](os.read(os.root / "tmp" / "config.json"))

      val spark = SparkSession
        .builder()
        .appName("Create Ratings Matrix")
        .config("spark.master", "yarn")
        .getOrCreate()

      spark.sparkContext.setLogLevel("ERROR")

      // Read files
      println(s"${stamp()} - start reading file")
      val ratingsByUser = (1 to 4)
        .map(n => spark.sparkContext
          .textFile(s"tabular_data_$n.txt")
          .map(l => {
            val splitted = l.split("\t")
            assert(splitted.length == 3)
            // movieIds are one indexed
            val movieId = splitted(0).toInt - 1
            val origUserId = splitted(1).toInt
            val rating = splitted(2).toDouble
            origUserId -> (Array(movieId), Array(rating))
        })).reduce((a,b) => a.union(b))
        .reduceByKey((a, b) => (a._1 ++ b._1) -> (a._2 ++ b._2))
        .map { case (uId, ratings) => {
          val ratingsZipped = ratings._1.zip(ratings._2)
          val sortedRatings = ratingsZipped.sortWith(_._1 < _._1)
          uId -> sortedRatings.unzip
        }}

      val maxMovieId = ratingsByUser.map {
          case (_, (moviesId, _)) => moviesId.max
        }.reduce(max(_,_))
      println(s"${stamp()} - maxMovieId: ${maxMovieId}")
      val uidConversionAndRows = ratingsByUser.map {
              case (orig_user_id: Int, 
                    (movie_ids: Array[Int],
                     ratings: Array[Double])) =>
                orig_user_id -> 
                  new SparseVector(maxMovieId + 1, movie_ids, ratings)
        }.zipWithIndex.map {
          case ((userId, vector), index) => 
            (userId, index) -> new IndexedRow(index, vector)
        }

      val uidConversion = uidConversionAndRows.map {case (a,b) => a}.collect().toMap
      val ratings = new IndexedRowMatrix(
        uidConversionAndRows.map {case (a,b) => b})
      println(s"${stamp()} - created ratings matrix")

      val matrixSize = ratings.numRows() * ratings.numCols()
      println(f"${stamp()} - rows: ${ratings.numRows}")
      println(f"${stamp()} - cols: ${ratings.numCols}")

      val interactions = ratings.rows
        .map(a => a.vector.numNonzeros).reduce(_+_)
      println(f"${stamp()} - interactions: ${interactions}")
      val sparsity = 100 * (interactions.toDouble / matrixSize.toDouble)

      println(f"${stamp()} - dimension: ${matrixSize}")
      println(f"${stamp()} - sparsity: ${sparsity}%.1f%%")

      // questa funzione non va in scala 2.12
      // showNotNull(ratings)

      println(s"${stamp()} - start train split")
      val trainSplit = config("trainSplit").floor.toInt
      val (train, test) = if (trainSplit > 0) {
          createTrainTest(ratings, trainSplit)
        } else {
          ratings -> readProbeDataset(
            spark.sparkContext, 
            ratingsByUser, 
            uidConversion
          )
        }
      train.rows.cache()
      test.entries.cache()

      val nFeatures = config("features").floor.toInt
      val nIters = config("iters").floor.toInt
      println(s"${stamp()} - nFeatures: $nFeatures")

      /*
      We use ALS to decompose ratings into two matrices of size
      U: features x users, M: features x movies
      TO DO THIS:
       * start with an M where the first row is the mean rating for that movie and the other values are random
       * send M to each node, distribute ratings by rows and calculate in a distributed way the U that reduces the error (euclidean distance)
       * collect U locally on each node
       * repeat the operation, but with ratings distributed by columns
      */

      val nItems = train.numCols().toInt
      val nUsers = train.numRows().toInt

      println(s"${stamp()} - start calc averages")
      val averages = train.toCoordinateMatrix.entries
        .map {
          case MatrixEntry(_, movie_id, rating) => movie_id -> (1, rating)
        }.foldByKey((0, 0))((movie1, movie2) =>
          (movie1._1 + movie2._1) -> (movie1._2 + movie2._2)
          // (movie_id, (n_ratings, sum_ratings))
        ).map {
          case (movieId, (nRatings, sumRatings)) =>
            movieId.toInt -> (sumRatings / nRatings)
        }.collect()

      // M0 is the M matrix at the first iteration
      println(s"${stamp()} - start create M0")
      val M0 =
        new Array[Double](
          nItems * nFeatures
        )
        averages.foreach { case (movie_id, avg_rating) =>
            M0(movie_id) = avg_rating
            for (feature <- 1 until nFeatures) {
              // the paper says "small random value", if it breaks in a weird
              // way this could be the reason
              val bigRate = config("bigRate")
              val ranDouble = Random.nextDouble()
              val bigPart = ((ranDouble * bigRate).floor / bigRate)
              M0(movie_id + feature.toInt) = ranDouble - bigPart
            }
        }
      val trainT = train.toCoordinateMatrix().transpose().toIndexedRowMatrix()
      trainT.rows.cache()


      // FOR CALC OF PREC@K AND REC@K
      val thresh = config("threshold")

      // Array[(user_id, Map[movie_id, is_relevant])]
      // we only sample the first N users, for performance
      // TODO: maybe we can increase this one
      val precRecallData = if (thresh > 0) {
          val sampledUsers = config("sampledUsersPrecRecall").floor.toInt
          val ks = Array(1,3,5,8,10,12,15)
          val testNotNull = test.toIndexedRowMatrix().rows.map(a => {
              a.index.toInt -> 
              a.vector.toArray.zipWithIndex.filter {
                case (rating, _) => rating != 0
              }.map {case (rating, index) => index -> (rating > thresh)}
              .toMap}).take(sampledUsers)
          Option(ks, testNotNull)
        } else {
          Option.empty
        }

      val testEntriesCount = test.entries.count().toDouble
      
      // DISPLAY SOME BASELINES
      // RMSE baseline
      val avrgsAsMap = averages.toMap
      def rmseBase(toWhat: CoordinateMatrix, toWhatSize: Double) = {
        sqrt(toWhat.entries.map {
          case MatrixEntry(_, movieId, rating) => 
            // println(s"movie average: ${avrgsAsMap(movieId.toInt)}, real rating: ${rating}")
            if (avrgsAsMap.contains(movieId.toInt)) {
              pow(avrgsAsMap(movieId.toInt) - rating, 2)
            } else {
              println(s"whats the average rating of movieId: ${movieId}?")
              pow(2.5 - rating, 2)
            }
        }.reduce(_+_) / toWhatSize)
      }
      println(f"${stamp()} - baseline, rms error to test set: ${rmseBase(test, testEntriesCount)}%.4f")
      val rmseTrainData = if (config("doRmseTrain") == 1.0) {
        val trainAsCoord = train.toCoordinateMatrix()
        Option(trainAsCoord, trainAsCoord.entries.count())
      } else {
        Option.empty
      }
      rmseTrainData.foreach {
        case (trainCoord, trainCoordCount) => {
          println(f"${stamp()} - baseline, rms error to train set: ${rmseBase(trainCoord, trainCoordCount)}%.4f")
        }
      }
      
      // PREC@K, REC@K BASELINE: RECOMMEND BASED ON AVERAGE
      precRecallData.foreach {
        case (ks, testNotNull) => {
          val reccOnAvgs = averages
            .filter { case (index, rating) => rating > thresh }
            .map { case (index, _) => index }

          // lot of memory wasted here,
          // dont care till it explodes since its done once
          val testNotNullAsMap = testNotNull.toMap
          val avrgMap = (0 until nUsers)
            .map(u => u -> 
              reccOnAvgs.filter(index => testNotNullAsMap.contains(u) && testNotNullAsMap(u).contains(index))
            ).toMap

          ks.map(k => {
            val (prec, rec) = precRecAtK(k, testNotNull, avrgMap, 100)
            println(f"${stamp()} - baseline, thresh: $thresh, prec@$k: ${prec}%.4f")
            println(f"${stamp()} - baseline, thresh: $thresh, rec@$k: ${rec}%.4f")
          })
        }
      }

      println(s"${stamp()} - start als")
      var M = new DenseMatrix(nFeatures, nItems, M0)
      var U = als_step(nFeatures, train, M)
      println(s"${stamp()} - done first half-iter")

      val lambda = config("lambda")

      (0 until nIters).map(iter => {
        U = als_step(lambda, train, M)
        M = als_step(lambda, trainT, U)
        println(s"${stamp()} - done iter: $iter")
        // error calculation
        // this could be done in parallel, if we didnt user mutable M, U
        // would save ~5s per iteration
        val rmseTest = rmse(M, U, test, testEntriesCount)
        println(f"${stamp()} - lambda: $lambda, iter: $iter, rms error to test set: $rmseTest%.4f")

        rmseTrainData.foreach {
          case (trainCoord, trainCoordCount) => {
            val rmseTrain = rmse(M, U, trainCoord, trainCoordCount)
            println(f"${stamp()} - lambda: $lambda, iter: $iter, rms error to train set: $rmseTrain%.4f")
          }
        }
        precRecallData.foreach {
          case (ks, testNotNull) => {
            val cachedK = cacheAtK(thresh, ks.max, M, U, testNotNull).toMap
            ks.foreach(k => {
              val (prec, rec) = precRecAtK(k, testNotNull, cachedK, 100)
              println(f"${stamp()} - iter: $iter, thresh: $thresh, prec@$k: ${prec}%.4f")
              println(f"${stamp()} - iter: $iter, thresh: $thresh, rec@$k: ${rec}%.4f")
            })
          }
        }
        println()
      })
    }

    def als_step(
        lambda: Double,            // regularization parameter
        ratings: IndexedRowMatrix, // ratings matrix
        from: DenseMatrix          // fixed matrix
    ) = {
      val nFeatures = from.numRows
      val to_unordered = ratings
        .rows
        .map(user => {
          val userId = user.index
          val userRatings = user.vector.toSparse

          //// invariant: nonZeroMovies is sorted in ascending order
          //val indexedNonZeroMovies = userRatings.toArray.zipWithIndex.filter {
          //  case (0, _) => false
          //  case (_, _) => true
          //} // only keep the movies that have been rated
          //assert(indexedNonZeroMovies.length > 0)

          //val nonZeroMoviesIds =
          //  indexedNonZeroMovies map { case (rating, index) =>
          //    index
          //  }
          //val nonZeroMoviesRatings =
          //  indexedNonZeroMovies map { case (rating, index) =>
          //    rating
          //  }

          val nonZeroMoviesIds = userRatings.indices
          val nonZeroMoviesRatings = userRatings.values

          // assert che sia column major
          assert(!from.isTransposed)
          /*
            directly use the `values` function on the dense matrix
            create a new dense matrix working directly on the underlying
            Double array

            What does it do? (example where `from` are the movies (M) and the rows of
            ratings are users)
            take, from the M matrix, only the movies (so the columns) that the
            user has reviewed
            Mm is this submatrix
           */
          val Mm_array =
            new Array[Double](nonZeroMoviesIds.length * nFeatures) // we remove the zero ratings because in the derivative formula they are not needed and we don't want to count them as actual values

          // for each movie that the user has rated
          // copy the movie from the M matrix to the Mm matrix
          assert(nonZeroMoviesIds.length != 0, s"why does user ${userId} have zero ratings?")

          var toMovieId = 0
          for (fromMovieId <- 0 to nonZeroMoviesIds.last) {
            if (fromMovieId == nonZeroMoviesIds(toMovieId)) { // POST: actually a foreach
              Array.copy( // we build the submatrix of movies that the user has rated
                from.values,
                fromMovieId * nFeatures,
                Mm_array,
                toMovieId * nFeatures,
                nFeatures
              )
              toMovieId += 1
            }
          }
          // all non zero movies should be in the new matrix
          assert(
            toMovieId == nonZeroMoviesIds.length,
            s"""current toMovieId is ${toMovieId}, 
              nonZeroMovies is 
              ${nonZeroMoviesIds.map(a => a.toString()).reduce(_ ++ "," ++ _)}"""
          )
          val Mm =
            new DenseMatrix(nFeatures, nonZeroMoviesIds.length, Mm_array)

          val tmp = Mm.multiply(Mm.transpose).values // Mm * Mm^T
          for (i <- 0 until nFeatures) {
            val j = i + (nFeatures * i)
            tmp(j) = tmp(j) + lambda * userRatings.numActives // n_(u i) * lambda
          }

          val V = Mm.multiply(new DenseVector(nonZeroMoviesRatings)).toArray
          userId -> gauss_method(nFeatures, tmp, V) // instead of calculating the inverse we use gauss (thx gauss)
          // }).sortByKey().map {case(i,v) => v}.reduce(_++_)
        }).collect()

      val ret_array = new Array[Double](ratings.numRows().toInt * nFeatures)
      // in questo ciclo faccio due operazioni:
      //  * ordino l'ouput dei vari nodi
      //  * riempio con zeri le colonne degli utenti di cui non so nulla.
      //    Per fare una cosa fatta bene servirebbe una cold-start strategy,
      //    ma sinceramente non mi interessa
      for (i <- 0 until to_unordered.length) { // we need the same ordering as in the rating matrix, but the collect doesn't give us the same order. we save the user index, so we can order for that
        Array.copy(
          to_unordered(i)._2,
          0,
          ret_array,
          to_unordered(i)._1.toInt * nFeatures,
          nFeatures
        )
      }

      new DenseMatrix(nFeatures, ratings.numRows.toInt, ret_array)
    }

    // lavora in-place su A,B (forse, sinceramente bho)
    def gauss_method(
        n: Integer,
        A: Array[Double],
        B: Array[Double]
    ): Array[Double] = {
      // For k = 1 : n Do:
      //   For i = 1 : n and if i! = k Do :
      //     piv := aik/akk
      //     For j := k + 1 : n + 1 Do :
      //       aij := aij − piv ∗ akj
      //     End
      //   End
      // End

      // questa operazione non vogliamo faccia una copia (va verificato)
      val C = A ++ B

      for (k <- 0 until n) {
        for (i <- 0 until n) {
          if (i != k) {
            // come facciamo se C_{k,k} è zero?
            assert(C(k + (k * n)) != 0.0)
            val piv = C(i + (k * n)) / C(k + (k * n))
            for (j <- k until (n + 1)) {
              val jn = j * n
              C(i + jn) = C(i + jn) - piv * C(k + jn)
            }
          }
        }
      }

      val ret = new Array[Double](n)
      for (i <- 0 until n) {
        ret(i) = C((n * n) + i) / C(i + (i * n))
      }

      ret
    }

    def createTrainTest(ratings: IndexedRowMatrix, nTest: Int) = {
      println("total entries: ", ratings
        .rows.map(a => a.vector.numActives).reduce(_+_))

      val splitted = ratings.rows.map {
        case IndexedRow(userId, row) => {
          val sparseRow = row.toSparse
          val actives = sparseRow.numActives
          val newNTest = if (actives <= nTest) {
            //println(s"user $userId has $actives ratings, cant take $nTest for testing")
            actives - 1
          } else {
            nTest
          }
          val nTrain = actives - newNTest
          assert(nTrain > 0)
          // wtf is wrong with you scala?
          val tmp = (sparseRow.indices zip sparseRow.values).toList
          val shuffledRow = Random.shuffle(tmp).toArray
          val trainRow = shuffledRow.take(nTrain).sortWith((t1, t2) => t1._1 < t2._1)
          val testRow = shuffledRow.takeRight(newNTest).sortWith((t1, t2) => t1._1 < t2._1)
          new IndexedRow(
            userId, 
            new SparseVector(
              row.size,
              trainRow.map(_._1),
              trainRow.map(_._2),
            )) -> (
              testRow.map {
                case (movieId, rating) =>
                  new MatrixEntry(userId.toLong, movieId.toLong, rating)
            })
        }
      }
      val train = new IndexedRowMatrix(splitted.map(_._1))
      val test = new CoordinateMatrix(
        splitted.map(_._2).flatMap(a => a))

      assert(ratings.numCols() == train.numCols())
      assert(ratings.numCols() == test.numCols())
      assert(ratings.numRows() == train.numRows())
      assert(ratings.numRows() == test.numRows())

      println("train entries: ", train
        .rows.map(a => a.vector.numActives).reduce(_+_))
      println("test entries: ", test.entries.count())

      train -> test
    }

    def readProbeDataset(
      sc: SparkContext, 
      fullDatasetByUser: RDD[(Int, (Array[Int], Array[Double]))],
      uidConversion: Map[Int, Long]
    ) = {
      val testFileParsed = sc.textFile("tabular_probe.txt")
          .map(l => {
            val splitted = l.split("\t")
            assert(splitted.length == 2)
            // movieIds are one indexed
            val movie_id = splitted(0).toInt - 1
            val orig_user_id = splitted(1).toInt
            orig_user_id -> movie_id
          })

      val testEntries = testFileParsed.join(fullDatasetByUser)
        .map { case (origUserId, (testMovieId, (movieIds, ratings))) => {
          val tmp = movieIds.indexOf(testMovieId)
          if (tmp == -1) {
            movieIds.foreach(a => println(origUserId, testMovieId, a))
          }

          new MatrixEntry(
            uidConversion(origUserId), 
            testMovieId.toLong,
            ratings(movieIds.indexOf(testMovieId))
          )
        }}
            
      new CoordinateMatrix(testEntries)
    }

    /* 
    def showNotNull(ratings: CoordinateMatrix) = {
      /*
       QUESTO FUNZIONA, CREA UN JFRAME
       val someData = 0 until 100 map (_ => Random.nextDouble() -> Random.nextDouble())
       val plot = xyplot(someData)(
                   par.withMain("Main label")
                   .withXLab("x axis label")
                   .withYLab("y axis label")
                 )
       val (frame, _) = show(plot)
       frame.setVisible(true)
      */

      // get non null entries for every user
      val notNullByUser = ratings
        .toRowMatrix()
        .rows
        .map(user =>
          user.numNonzeros.toDouble // number of movies rated by the user
        )
        .collect()
        .sorted
        .zipWithIndex
        .map {
          case (a, b) => b.toDouble -> a
        }.toList

        val plot8 = xyplot(
          notNullByUser -> bar(
          horizontal = false,
          width = 0.1,
          fill = Color.gray2
        )
        )(
          par.xlab("x axis label")
             .ylab("y axis label")
             .ylog(true)
             .xlim(Some(0d -> 1000d))
        )

        val (frame, _) = show(plot8)
        frame.setVisible(true)
    }
    */
    def rmse(
      M: DenseMatrix,
      U: DenseMatrix,
      R: CoordinateMatrix,
      Rcount: Double
    ) = {
      val nFeatures = M.numRows
      val Uvals = U.values
      val Mvals = M.values
      sqrt(R.entries.map {
        case MatrixEntry(user_id, movie_id, rating) => {
          // perform row-column multiplication
          val expected = (0 until nFeatures).map(i =>
            Uvals(user_id.toInt * nFeatures + i)
              * Mvals(movie_id.toInt * nFeatures + i)
          ).reduce(_ + _)
          pow(expected - rating, 2)
        }
      }.reduce(_ + _) / Rcount)
    }

    def cacheAtK (
      thresh: Double,
      maxK: Integer,
      M: DenseMatrix,
      U: DenseMatrix,
      testCached: Array[(Int, Map[Int,Boolean])],
    ) = {
      testCached.map { case (index, row) => {
        val userFeatures = U.colIter.drop(index).next
        // recRow = Array[(computedRating, movie_id)]
        val recRow = M.colIter
          .zipWithIndex.filter {
            // consideriamo solo i film contenuti nella matrice di test
            case (_, index) => row.keySet.contains(index)
          }.map(a => a._1.dot(userFeatures) -> a._2).toArray

        // ret = (user_id, Array[movie_id])
        // questi sono i film recommanded
        index -> recRow.filter {
          case (rating, _) => rating >= thresh
        }.sortWith((a, b) => a._1 > b._1).take(maxK).map {
          case (_, index) => index
        }}
      }
    }

    def precRecAtK (
        k: Integer,
        testCached: Array[(Int, Map[Int,Boolean])],
        suggestedCached: Map[Int, Array[Int]],
        stopAt: Int
    ) = {
      val (prec, rec) = testCached.map { case (userId, movies) => {
        // k is possibly bigger than the number of recommanded movies
        val suggestedForUser = suggestedCached(userId).take(k)

        // relevantRecomm are the movies suggested and actually relevant
        val relevantRecomm = suggestedForUser
          .filter(index => movies(index)).length

        val precK = if (suggestedForUser.length > 0) {
          relevantRecomm.toDouble / suggestedForUser.length.toDouble
        } else {
          1.0
        }

        val allRelevant = movies.filter(a => a._2).size
        val recK = if (allRelevant > 0) {
          relevantRecomm.toDouble / allRelevant.toDouble
        } else {
          1.0
        }
        precK -> recK
      }}.reduce((a, b) => a._1 + b._1 -> (a._2 + b._2))

      val counted = testCached.length
      prec / counted -> (rec / counted)
    }
  }

}
