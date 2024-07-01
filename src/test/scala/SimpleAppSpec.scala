import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers 
import org.apache.spark.mllib.linalg.{Matrices, Matrix, DenseVector, DenseMatrix}
import org.apache.spark.mllib.linalg.distributed.{
  CoordinateMatrix,
  MatrixEntry,
  RowMatrix,
  BlockMatrix
}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import SimpleApp.SimpleApp.U_als_step

class UAlsStepSpec extends AnyFlatSpec with Matchers  {

    "U_als_step" should "return the expected DenseMatrix" in {

        // Define Spark configuration
        val conf = new SparkConf().setAppName("SimpleApp").setMaster("local[*]")

        // Initialize SparkContext
        val sc = new SparkContext(conf)
          
        val ratings = new CoordinateMatrix(sc.parallelize(Seq(
            (0L, 0L, 1.0),
            (1L, 1L, 2.0),
            (2L, 2L, 3.0)
        ).map { case (i, j, value) => new MatrixEntry(i, j, value) }))
        val M: DenseMatrix = Matrices.dense(3, 3, Array(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)).asInstanceOf[DenseMatrix]         
        val features: Integer = M.numCols.toInt
        val numUsers: Integer = ratings.numRows.toInt
        val lambda = 10
        
        val expectedMatrix = Matrices.dense(3, 3, Array[Double](1/11.0, 0.0, 0.0, 0.0, 2/11.0, 0.0, 0.0, 0.0, 3/11.0))
        val resultMatrix = U_als_step(features, numUsers, lambda, ratings, M)

        resultMatrix shouldEqual expectedMatrix
    }
}