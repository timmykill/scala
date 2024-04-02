import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}



object SimpleApp {
	def main(args: Array[String]): Unit = {
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
			MatrixEntry(userId - 1, itemId - 1, rating)})

		val coordMatrix = new CoordinateMatrix(entries)
		val matrix_size : Double = coordMatrix.numRows() * coordMatrix.numCols()
		println(f"rows: ${coordMatrix.numRows()}")
		println(f"cols: ${coordMatrix.numCols()}")
		val interactions = entries.count()
		println(f"interactions: ${interactions}")
		val sparsity = 100 * (interactions / matrix_size)

		println(f"dimension: ${matrix_size}")
		println(f"sparsity: $sparsity%.1f%%")


		//println("HERE")
		//ratingsRDD.collect().foreach(println)

		}
}
