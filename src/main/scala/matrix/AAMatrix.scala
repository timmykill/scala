package matrix

import matrix.MatrixExceptions._

case class AAMatrix(data: Array[Array[Double]]) {
  def rows: Int = data.length
  def cols: Int = if (data.isEmpty) 0 else data(0).length

  def multiply(that: AAMatrix): AAMatrix = {
	if (this.cols != that.rows) {
	  throw new WrongSizeException
	} else {
	  val result = Array.ofDim[Double](this.rows, that.cols)
	  for (i <- 0 until this.rows) {
		for (j <- 0 until that.cols) {
		  for (k <- 0 until this.cols) {
			result(i)(j) += this.data(i)(k) * that.data(k)(j)
		  }
		}
	  }
	  AAMatrix(result)
	}
  }

  override def toString: String = data.map(_.mkString(" ")).mkString("\n")
}

object AAMatrix {
	def apply(rows: Int, cols: Int, values: Double*): AAMatrix = {
		require(values.length == rows * cols, "Invalid data length")
		AAMatrix(values.toArray.grouped(cols).toArray)
	}
}
