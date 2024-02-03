case class Matrix(data: Array[Array[Double]]) {
  def rows: Int = data.length
  def cols: Int = if (data.isEmpty) 0 else data(0).length

  def multiply(that: Matrix): Option[Matrix] = {
	if (this.cols != that.rows) {
	  None
	} else {
	  val result = Array.ofDim[Double](this.rows, that.cols)
	  for (i <- 0 until this.rows) {
		for (j <- 0 until that.cols) {
		  for (k <- 0 until this.cols) {
			result(i)(j) += this.data(i)(k) * that.data(k)(j)
		  }
		}
	  }
	  Some(new Matrix(result))
	}
  }

  override def toString: String = data.map(_.mkString(" ")).mkString("\n")
}

object Matrix {
	def apply(rows: Int, cols: Int, values: Double*): Matrix = {
		require(values.length == rows * cols, "Invalid data length")
		Matrix(values.toArray.grouped(cols).toArray)
	}
}

object SimpleApp {
	def main(args: Array[String]): Unit = {
		val matrixA = Matrix(Array(
				Array(1, 2, 3),
				Array(4, 5, 6)
			)
		)

		val matrixB = Matrix(Array(
				Array(7, 8),
				Array(9, 10),
				Array(11, 12)
			)
		)

		println("Matrix A:")
		println(matrixA)
		println("Matrix B:")
		println(matrixB)

		val result = matrixA.multiply(matrixB)

		println("Matrix A * Matrix B:")
		result match {
			case Some(matrix) => println(matrix)
			case None => println("Invalid matrix dimensions")
		}
  }
}
