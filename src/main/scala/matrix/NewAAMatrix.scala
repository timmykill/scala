package matrix

import matrix.MatrixExceptions._

case class NewAAMatrix(cols: Int, data: Array[Array[Double]]) {
	if (cols != data.length) { throw new WrongSizeException }
	val rowLengths = data.map(a => a.length).toSet
	if (rowLengths.size != 1) { throw new WrongSizeException }
	def rows: Int = rowLengths.head

	def multiply(that: NewAAMatrix) = {
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
			NewAAMatrix(that.cols, result)
		}
	}

	override def toString: String = data.map(_.mkString(" ")).mkString("\n")
}
