package matrix

import matrix.MatrixExceptions._

case class AMatrix(val cols: Int, val data: Array[Double]) {
	val dataLength = data.length
	if (dataLength % cols != 0) { throw new WrongSizeException }
	val rows = dataLength / cols

	def rowsIterator = data.grouped(cols)
	def colsIterator = (0 until cols).map(a => { 
		for {
			(b, i) <- data.zipWithIndex
			if i % cols ==  a
		} yield b })

	def multiply(that: AMatrix) = {
		if (cols != that.rows) {
			throw new WrongSizeException
		} else {
			val newArr = rowsIterator.flatMap(
			    (a: Array[Double]) => that.colsIterator
			        .map((b: Array[Double]) => a.zip(b)
			            .map((c: (Double, Double)) => c._1 * c._2)
			            .fold(0.0)(_+_)))
			AMatrix(that.cols, newArr.toArray)
		}
	}

	override val toString : String = rowsIterator.map(a => a.mkString(" "))
	                                             .mkString("\n")

}
