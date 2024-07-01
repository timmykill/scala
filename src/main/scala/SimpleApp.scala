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

object SimpleApp {

  val spark = SparkSession
    .builder()
    .appName("Create Ratings Matrix")
    .config("spark.master", "local")
    .getOrCreate()

  def create_train_test(ratings: CoordinateMatrix) = {
    // TODO: questo può essere sicuramente fatto anche senza passare per IndexedRowMatrix
    val trainIndexes = ratings
      .toIndexedRowMatrix()
      .rows
      .map(a => {
        val i = a.index
        val f = a.vector.toArray.toList.zipWithIndex
          .filter(b => b._1 != 0)
          .map(b => b._2)
        i -> Random.shuffle(f).take(10)
      })
      .collect()
      .toMap

    val train = new CoordinateMatrix(
      ratings.entries
        .filter(a => trainIndexes(a.i).contains(a.j))
    )

    val test = new CoordinateMatrix(
      ratings.entries
        .filter(a => !trainIndexes(a.i).contains(a.j))
    )

    // assert that train and test are disjoint
    assert(
      train.entries
        .map(a => (a.i, a.j))
        .intersection(
          test.entries
            .map(a => (a.i, a.j))
        )
        .count() == 0
    )

    println("the assert is true!")
    println(train.entries.count())
    println(test.entries.count())

    train -> test
  }

  def predict(user_factors: DenseVector, item_factors: DenseVector): Double = {
    user_factors.dot(item_factors)
  }

  def U_als_step(features: Integer, users: Integer, lambda: Integer, ratings: CoordinateMatrix, M: DenseMatrix) = {
    val U_arr = ratings.toIndexedRowMatrix().rows.map(index_row => {
      val index = index_row.index
      val row = index_row.vector

      // invariante: non_zero_movies è crescente
      val non_zero_movies= row.toArray.zipWithIndex.filter {
        case (0, _) => false
        case (_, _) => true
      }     
      // print("non_zero_movies: ")
      // non_zero_movies.foreach {
      //   case (v, i) => print(s"($i, $v) ")
      // }
      // println()
      val non_zero_movies_index = non_zero_movies map {
        case (v, i) => i
      }
      val non_zero_movies_values = non_zero_movies map {
        case (v, i) => v
      }


      // assert che sia column major
      assert(!M.isTransposed)
      // usa direttamente la funzione `values` sulla dense matrix
      // crea una nuova dense matrix lavorando direttamente sull'array di 
      // Double sottostante
      val Mm_array = new Array[Double](non_zero_movies_index.length * features)
      var to_i = 0
      for (from_i <- 0 to non_zero_movies_index.last) {
        if (from_i == non_zero_movies_index(to_i)) {
           Array.copy(M.values,
                       from_i * features,
                       Mm_array,
                       to_i * features,
                       features)
           to_i += 1
        }
      }
      // all non zero movies should be in the new matrix
      assert(to_i == non_zero_movies_index.length)
      val Mm = new DenseMatrix(features, non_zero_movies_index.length, Mm_array)

      val tmp = Mm.multiply(Mm.transpose).values
      // Avere solo il "triangolo superiore" di A può essere utile per
      // utilizzare il metodo di Cholesky
      // val A_sup = new Array[Double]((features * (features + 1)) / 2)
      // for (i <- 0 until features) {
      //   Array.copy(tmp, (i * features), A_sup, (i*(i+1))/2, i)
      //   A_sup(((i*(i+1))/2)-1) = tmp((i * features) + i) + lambda * non_zero_movies.length
      // }
      //
      for (i <- 0 until features) {
        val j = i + (features * i)
        tmp(j) = tmp(j) + lambda * non_zero_movies.length
      }

      val V = Mm.multiply(new DenseVector(non_zero_movies_values)).toArray
      gauss_method(features, tmp, V)
    }).reduce(_++_)
    new DenseMatrix(features, users, U_arr)
  }

  // lavora in-place su A,B (forse, sinceramente bho)
  def gauss_method(n: Integer, A: Array[Double], B: Array[Double]): Array[Double] = {
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
        if (i != k){
          // come facciamo se C_{k,k} è zero?
          val piv = C(i + (k * n)) / C(k + (k *n))
          for (j <- k until (n + 1)) {
            val jn = j * n
            C(i + jn) = C(i + jn) - piv * C(k + jn)
          }
        }
      }
    }

    val ret = new Array[Double](n)
    for (i <- 0 until n) { 
      ret(i) = C(i + (i * n)) / C((n * n) + i)
    }

    ret
  }

  def main(args: Array[String]): Unit = {

    // QUESTO FUNZIONA, CREA UN JFRAME
    // val someData = 0 until 100 map (_ => Random.nextDouble() -> Random.nextDouble())
    // val plot = xyplot(someData)(
    //             par.withMain("Main label")
    //             .withXLab("x axis label")
    //             .withYLab("y axis label")
    //           )
    // val (frame, _) = show(plot)
    // frame.setVisible(true)

    spark.sparkContext.setLogLevel("ERROR")

    // colonne: user_id item_id rating timestamp
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

    val nUsers = df
      .select(countDistinct("user_id"))
      .collect()(0)(0)
      .asInstanceOf[Number]
      .intValue()
    val nItems = df
      .select(countDistinct("item_id"))
      .collect()(0)(0)
      .asInstanceOf[Number]
      .intValue()

    val entries = df.rdd.map(row => {
      val userId = row.getAs[Int]("user_id")
      val itemId = row.getAs[Int]("item_id")
      val rating = row.getAs[Double]("rating")
      MatrixEntry(userId - 1, itemId - 1, rating)
    })

    // CoordinateMatrix è una matrice distribuita
    // utilizza COO per memorizzarla (in pratica è una lista di entry)
    val ratings = new CoordinateMatrix(entries)

    val matrix_size: Double = ratings.numRows() * ratings.numCols()
    println(f"rows: ${ratings.numRows()}")
    println(f"cols: ${ratings.numCols()}")

    val interactions = entries.count()
    println(f"interactions: ${interactions}")
    val sparsity = 100 * (interactions / matrix_size)

    println(f"dimension: ${matrix_size}")
    println(f"sparsity: $sparsity%.1f%%")

    // get non null entries for every user
    val notNullByUser = ratings
      .toRowMatrix()
      .rows
      .map(a => a.numNonzeros.toDouble)
      .collect()
      .sorted(Ordering.Double.IeeeOrdering.reverse)
      .zipWithIndex
      .map(a => a._2.toDouble -> a._1)
      .toList
    val barPlotData = List(1 -> 10, 2 -> 11)

    val plot8 = xyplot(
      notNullByUser -> bar(horizontal = false, width = 0.1, fill = Color.gray2)
    )(
      par
        .xlab("x axis label")
        .ylab("y axis label")
        .ylog(true)
        .xlim(Some(0d -> 1000d))
    )
    val (frame, _) = show(plot8)
    frame.setVisible(true)

    val (train, test) = create_train_test(ratings)

    println("HERE")
    println(train.toString())

    // println("HERE")
    // ratingsRDD.collect().foreach(println)

    val features = 10
    val n_iters = 100

    // Ora dobbiamo usare ALS per decomporre ratings in due matrici di
    // dimensione U: features x users, M: features x movies
    // PER FARLO:
    //  * partiamo con un M in cui la prima riga corrisponde alla media del
    //    rating per quel film e gli altri valori sono random
    //  * mandiamo M ad ogni nodo, distribuiamo ratings per righe e calcoliamo
    //    in modo distribuito la U che riduce l'errore (distanza euclidea)
    //  * collezioniamo U in locale su ogni nodo
    //  * ripetiamo l'operazione, ma con ratings distribuito per colonne
    
    val first_M_array = new Array[Double](nItems * features)
    ratings.entries.map(a => a.j -> (1, a.value))
                     .foldByKey((0,0))((a, b) => (a._1 + b._1) -> (a._2 + b._2))
                     .map(a => a._1 -> (a._2._2 / a._2._1))
                     .collect().foreach {
                       case (i1, v) => first_M_array(i1.toInt) = v
                                      for (i2 <- 1 until features) {
        // il paper qui dice "small random value", se si rompe in modo strano
        // questo potrebbe essere il motivo
                                        first_M_array(i1.toInt + i2.toInt) = Random.nextDouble()
                                      }}
    val M = new DenseMatrix(features, nItems, first_M_array)
    val U = U_als_step(features, nUsers, 10, ratings, M)

    println(U)

  }
}
