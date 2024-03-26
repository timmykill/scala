// IMPLEMENTIAMO https://ethen8181.github.io/machine-learning/recsys/1_ALSWR.html

import org.apache.spark.mllib.linalg.{Matrix, Matrices, DenseMatrix}
import matrix.AAMatrix



object SimpleApp {
	def main(args: Array[String]): Unit = {

//		ESEMPIO CREAZIONE MATRICI
//		// Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
//		val dm: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))
//
//		// Create a sparse matrix ((9.0, 0.0), (0.0, 8.0), (0.0, 6.0))
//		val sm: Matrix = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 2, 1), Array(9, 6, 8))

		val test = scala.io.Source.fromFile("ml-100k/u.data").getLines.toArray.flatMap(_.split("\t")).map(_.toDouble)
		// colonne: user_id item_id rating timestamp
		//val m1 = new DenseMatrix(100000,4,test,true)
		val m1 = AAMatrix(100000,4,test)
		val n_unique_users = m1.rowIter.map(_(0)).distinct.length
		val n_unique_items = m1.rowIter.map(_(1)).distinct.length

		val ratings = AAMatrix.zeros(n_unique_users, n_unique_items)

		m1.rowIter.foreach(a => {
			ratings.update(a(0).toInt -1, a(1).toInt -1, a(2));
		})

		val matrix_size : Double = n_unique_users * n_unique_items
		val interaction : Double = ratings.flatNonZero.length
		val sparsity = 100 * (interaction / matrix_size)

		println(f"dimension: ${ratings.shape}")
		println(f"sparsity: $sparsity%.1f%%")


	}
}
