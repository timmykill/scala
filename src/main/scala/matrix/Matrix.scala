package matrix

trait Matrix {
	def multiply: Matrix => Matrix
	def sum: Matrix => Matrix
	override def toString: String
}


object MatrixExceptions {
	class WrongSizeException extends RuntimeException
}
