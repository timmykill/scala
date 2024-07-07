import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._
import org.apache.spark.mllib.linalg.distributed.{
  CoordinateMatrix,
  MatrixEntry,
  RowMatrix,
  BlockMatrix
}
import org.apache.spark.mllib.linalg.{DenseVector, Matrices, DenseMatrix}
import org.nspl._
import org.nspl.awtrenderer._
import scala.util.Random
import java.awt.Panel
import java.awt.Frame
import java.awt.Graphics
import javax.imageio.ImageIO
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.CholeskyDecomposition

package SimpleApp {
  object SimpleApp {

    val spark = SparkSession
      .builder()
      .appName("Create Ratings Matrix")
      .config("spark.master", "local")
      .getOrCreate()

    def create_train_test(ratings: CoordinateMatrix) = {
      // TODO: this can be probably done without using IndexedRowMatrix
      val trainIndexes = ratings
        .toIndexedRowMatrix()
        .rows
        .map(user => {
          val userId = user.index
          val ratedMovies = user.vector.toArray.toList.zipWithIndex
            .filter(indexedRating =>
              indexedRating._1 != 0 // filter out the movies that have not been rated by the user
            )
            .map(indexedNonZeroRating =>
              indexedNonZeroRating._2 // keep only the indexes of the movies
            )
          // never take the last
          // TODO: why?
          userId -> Random.shuffle(ratedMovies).take(10)
        })
        .collect()
        .toMap

      val train = new CoordinateMatrix(
        ratings.entries
          .filter(entry => trainIndexes(entry.i).contains(entry.j)),
        ratings.numRows(),
        ratings.numCols()
      )

      val test = new CoordinateMatrix(
        ratings.entries
          .filter(a => !trainIndexes(a.i).contains(a.j)),
        ratings.numRows(),
        ratings.numCols()
      )

      // assert that train and test are disjoint
      assert(
        train.entries
          .map(trainEntry => (trainEntry.i, trainEntry.j))
          .intersection(
            test.entries
              .map(testEntry => (testEntry.i, testEntry.j))
          )
          .count() == 0
      )

      println("ratings:", ratings.numRows(), "x", ratings.numCols())
      println("train:", train.numRows(), "x", train.numCols())
      println("test:", test.numRows(), "x", test.numCols())
      assert(ratings.numCols() == train.numCols())
      assert(ratings.numCols() == test.numCols())
      assert(ratings.numRows() == train.numRows())
      assert(ratings.numRows() == test.numRows())

      println("the asserts are true!")
      println("train entries: ", train.entries.count())
      println("test entries: ", test.entries.count())

      train -> test
    }

    def als_step(
        lambda: Integer, // regularization parameter
        ratings: CoordinateMatrix, // ratings matrix
        from: DenseMatrix // fixed matrix
    ) = {
      val nFeatures = from.numRows
      val to_unordered = ratings
        .toIndexedRowMatrix()
        .rows
        .map(user => {
          val userId = user.index
          val userRatings = user.vector

          // invariant: nonZeroMovies is sorted in ascending order
          val indexedNonZeroMovies = userRatings.toArray.zipWithIndex.filter {
            case (0, _) => false
            case (_, _) => true
          } // only keep the movies that have been rated
          assert(indexedNonZeroMovies.length > 0)

          val nonZeroMoviesIds =
            indexedNonZeroMovies map { case (rating, index) =>
              index
            }
          val nonZeroMoviesRatings =
            indexedNonZeroMovies map { case (rating, index) =>
              rating
            }

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
            new Array[Double](nonZeroMoviesIds.length * nFeatures)

          // for each movie that the user has rated
          // copy the movie from the M matrix to the Mm matrix
          var toMovieId = 0
          for (fromMovieId <- 0 to nonZeroMoviesIds.last) {
            if (fromMovieId == nonZeroMoviesIds(toMovieId)) {
              Array.copy(
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
          assert(toMovieId == nonZeroMoviesIds.length)
          val Mm =
            new DenseMatrix(nFeatures, nonZeroMoviesIds.length, Mm_array)

          val tmp = Mm.multiply(Mm.transpose).values // Mm * Mm^T
          for (i <- 0 until nFeatures) {
            val j = i + (nFeatures * i)
            tmp(j) = tmp(j) + lambda * indexedNonZeroMovies.length
          }

          val V = Mm.multiply(new DenseVector(nonZeroMoviesRatings)).toArray
          userId -> gauss_method(nFeatures, tmp, V)
          // }).sortByKey().map {case(i,v) => v}.reduce(_++_)
        })
        .collect()

      val ret_array = new Array[Double](ratings.numRows().toInt * nFeatures)
      // in questo ciclo faccio due operazioni:
      //  * ordino l'ouput dei vari nodi
      //  * riempio con zeri le colonne degli utenti di cui non so nulla.
      //    Per fare una cosa fatta bene servirebbe una cold-start strategy,
      //    ma sinceramente non mi interessa
      for (i <- 0 until to_unordered.length) {
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

    def main(args: Array[String]): Unit = {

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

      spark.sparkContext.setLogLevel("ERROR")

      // columns: user_id item_id rating timestamp
      val customSchema = StructType(
        Array(
          StructField("user_id", IntegerType, true),
          StructField("item_id", IntegerType, true),
          StructField("rating", DoubleType, true),
          StructField("timestamp", LongType, true)
        )
      )

      // Read the CSV file into a DataFrame
      val df = spark.read
        .option("header", "true")
        .schema(customSchema)
        .option("delimiter", "\t")
        .csv("ml-100k/u.data")

      df.show()

      // list of matrix entries needed to create the coordinate matrix
      val entries = df.rdd.map(row => {
        val userId = row.getAs[Int]("user_id")
        val itemId = row.getAs[Int]("item_id")
        val rating = row.getAs[Double]("rating")
        MatrixEntry(userId - 1, itemId - 1, rating) // row, column, value
      })

      // CoordinateMatrix is a distributed matrix which
      // uses COO as the underlying storage (which is, in practice, a list of matrix entries)
      // ideally sorted first by row, then by column
      val ratings = new CoordinateMatrix(entries)

      val matrixSize: Double = ratings.numRows() * ratings.numCols()
      println(f"rows: ${ratings.numRows()}")
      println(f"cols: ${ratings.numCols()}")

      val interactions = entries.count()
      println(f"interactions: ${interactions}")
      val sparsity = 100 * (interactions / matrixSize)

      println(f"dimension: ${matrixSize}")
      println(f"sparsity: $sparsity%.1f%%")

      /*
      // get non null entries for every user
      val notNullByUser = ratings
        .toRowMatrix()
        .rows
        .map(user =>
          user.numNonzeros.toDouble // number of movies rated by the user
        )
        .collect()
        .sorted(
          Ordering.Double.IeeeOrdering.reverse // sort in descending order to get the most active users first
        )
        .zipWithIndex
        .map(a => a._2.toDouble -> a._1 // from (value, index) to (index, value)
        )
        .toList

      val plot8 = xyplot(
        notNullByUser -> bar(
          horizontal = false,
          width = 0.1,
          fill = Color.gray2
        )
      )(
        par
          .xlab("x axis label")
          .ylab("y axis label")
          .ylog(true)
          .xlim(Some(0d -> 1000d))
      )
      val (frame, _) = show(plot8)
      // // don't show it every time
      frame.setVisible(true)
       */

      val (train, test) = create_train_test(ratings)

      val nFeatures = 10
      val nIters = 10

      /*
      We use ALS to decompose ratings into two matrices of size
      U: features x users, M: features x movies
      TO DO THIS:
       * start with an M where the first row is the mean rating for that movie and the other values are random
       * send M to each node, distribute ratings by rows and calculate in a distributed way the U that reduces the error (euclidean distance)
       * collect U locally on each node
       * repeat the operation, but with ratings distributed by columns
       */

      val nItems = df
        .select(countDistinct("item_id"))
        .collect()(0)(0)
        .asInstanceOf[Number]
        .intValue()

      // M0 is the M matrix at the first iteration
      val M0 =
        new Array[Double](
          nItems * nFeatures
        )

      train.entries
        .map(entry => entry.j -> (1, entry.value)) // (movie_id, (1, rating))
        .foldByKey((0, 0))((movie1, movie2) =>
          (movie1._1 + movie2._1) -> (movie1._2 + movie2._2) // (movie_id, (n_ratings, sum_ratings))
        )
        .map(movie =>
          movie._1 -> (movie._2._2 / movie._2._1) // (movie_id, avg_rating)
        )
        .collect()
        .foreach { case (movie_id, avg_rating) =>
          M0(movie_id.toInt) = avg_rating
          for (feature <- 1 until nFeatures) {
            // the paper says "small random value", if it breaks in a weird way
            // this could be the reason
            M0(movie_id.toInt + feature.toInt) = Random.nextDouble()
          }
        }
      var M = new DenseMatrix(nFeatures, nItems, M0)
      var U = als_step(nFeatures, train, M)
      val trainT = train.transpose()
      for (iter <- 0 until nIters) {
        U = als_step(nFeatures, train, M)
        M = als_step(nFeatures, trainT, U)
        // error calculation
        val err = test.entries
          .map {
            case MatrixEntry(user_id, movie_id, _) => {
              val Uvals = U.values
              val Mvals = M.values
              (for (feature <- 0 until nFeatures)
                yield Uvals(user_id.toInt * nFeatures + feature)
                  * Mvals(movie_id.toInt * nFeatures + feature))
                .reduce(_ + _)
            }
          }
          .reduce(_ + _)
        println(s"iter: $iter, error: $err")
      }
    }
  }
}
