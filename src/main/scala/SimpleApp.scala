import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.nspl._
import org.nspl.awtrenderer._
import scala.util.Random
import java.awt.Panel
import java.awt.Frame
import java.awt.Graphics
import javax.imageio.ImageIO


object SimpleApp {
	def create_train_test(ratings: CoordinateMatrix) = {
		val trainIndexes = ratings.toIndexedRowMatrix().rows
		                          .map(a => {val i = a.index
		                                     val f = a.vector.toArray.toList
		                                              .zipWithIndex
		                                              .filter(b => b._2 != 0)
		                                              .map(b => b._1)
		                                     i -> Random.shuffle(f).take(10)})
		                          .collect()
		                          .toMap
		val train = ratings.toIndexedRowMatrix().rows
		                   .map(a => {val x = a.vector.toArray.toList
		                                       .zipWithIndex
		                                       .map(b => {if (trainIndexes(a.index).contains(b._1)) {
		                                                     b._1
		                                                  } else {
		                                                     0 }}).toArray
		                              val y = DenseVector(x)
		                              a.index -> y})

		                          testIndexes(a.index).contains(a.j)))
		val test = new CoordinateMatrix(ratings.entries
		                  .filter(a => !testIndexes(a.i).contains(a.j)))

		assert(train.toRowMatrix()
		            .multiply(test.transpose().toBlockMatrix().toLocalMatrix())
		            .rows.map(a => a.numNonzeros)
		            .filter(a => a != 0).count() == 0)

		train -> test
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

		val spark = SparkSession.builder()
		  .appName("Create Ratings Matrix")
		  .config("spark.master", "local")
		  .getOrCreate()
		spark.sparkContext.setLogLevel("ERROR")

		// colonne: user_id item_id rating timestamp
		val customSchema = StructType(Array(
		  StructField("user_id", IntegerType, true),
		  StructField("item_id", IntegerType, true),
		  StructField("rating", DoubleType, true),
		  StructField("timestamp", LongType, true))
		)

		// Read the CSV file into a DataFrame
		val df = spark.read
		  .option("header", "true")
		  .schema(customSchema)
		  .option("delimiter", "\t")
		  .csv("ml-100k/u.data")

		df.show()

		val nUsers = df.select(countDistinct("user_id")).collect()(0)(0).asInstanceOf[Number].intValue()
		val nItems = df.select(countDistinct("item_id")).collect()(0)(0).asInstanceOf[Number].intValue()

		//val bigArray = Array(nUsers * nItems)
		//ratings.map(row => {
		//	case (userId, itemId, rating, _) => {
		//		bigArray(userId * nUsers + itemId) = rating}})

		val entries = df.rdd.map(row => {
			val userId = row.getAs[Int]("user_id")
			val itemId = row.getAs[Int]("item_id")
			val rating = row.getAs[Double]("rating")
			MatrixEntry(userId - 1, itemId - 1, rating)
		})

		val ratings = new CoordinateMatrix(entries)
		val matrix_size : Double = ratings.numRows() * ratings.numCols()
		println(f"rows: ${ratings.numRows()}")
		println(f"cols: ${ratings.numCols()}")
		val interactions = entries.count()
		println(f"interactions: ${interactions}")
		val sparsity = 100 * (interactions / matrix_size)

		println(f"dimension: ${matrix_size}")
		println(f"sparsity: $sparsity%.1f%%")

		//get non null entries for every user
		val notNullByUser = ratings.toRowMatrix().rows
		           .map(a => a.numNonzeros.toDouble)
		           .collect()
		           .sorted(Ordering.Double.IeeeOrdering.reverse).zipWithIndex
		           .map(a => a._2.toDouble -> a._1).toList
		val barPlotData = List(1 -> 10, 2 -> 11)
		
		val plot8 = xyplot(notNullByUser -> bar(horizontal=false, 
		                                    width = 0.1, 
		                                    fill = Color.gray2)
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

		//println("HERE")
		//ratingsRDD.collect().foreach(println)

		}
}
