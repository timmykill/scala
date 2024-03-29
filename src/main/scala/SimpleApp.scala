import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS


case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)

object Rating {
	def parseRating(str: String): Rating = {
	  val fields = str.split("::")
	  assert(fields.size == 4)
	  Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
	} 
}

object SimpleApp {
	def main(args: Array[String]): Unit = {
		val logFile = "aaa.txt" // Should be some file on your system
    	val spark = SparkSession.builder.appName("Simple Application")
	                                .config("spark.master", "local")
	                                .getOrCreate()
		import spark.implicits._

		val ratings = spark.read.textFile("sample_movielens_ratings.txt")
		  .map(Rating.parseRating)
		  .toDF()
		val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

		// Build the recommendation model using ALS on the training data
		val als = new ALS()
		  .setMaxIter(5)
		  .setRegParam(0.01)
		  .setUserCol("userId")
		  .setItemCol("movieId")
		  .setRatingCol("rating")
		val model = als.fit(training)

		// Evaluate the model by computing the RMSE on the test data
		// Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
		model.setColdStartStrategy("drop")
		val predictions = model.transform(test)

		val evaluator = new RegressionEvaluator()
		  .setMetricName("rmse")
		  .setLabelCol("rating")
		  .setPredictionCol("prediction")
		val rmse = evaluator.evaluate(predictions)
		println(s"Root-mean-square error = $rmse")

		// Generate top 10 movie recommendations for each user
		val userRecs = model.recommendForAllUsers(10)
		// Generate top 10 user recommendations for each movie
		val movieRecs = model.recommendForAllItems(10)
		spark.stop()
  }
}
