package matrix

import matrix.MatrixExceptions._

case class AAMatrix(data: Array[Array[Double]]) {
  def rows: Int = data.length
  def cols: Int = if (data.isEmpty) 0 else data(0).length

  // per compatibilità con rowIter spark
  def rowIter: Iterator[Array[Double]] = data.iterator
  
  def shape: (Int, Int) = (rows, cols)

  def flatNonZero: List[Double] = {
	var result = List[Double]()
		for (i <- 0 until this.rows) {
			for (j <- 0 until this.cols) {
				val cell = data(i)(j);
				if (cell != 0.0) {
				  result = cell +: result
				}
			}
		}
	result
  }

  def update(i: Int, j: Int, v: Double): Unit = {
	  if (i >= rows || j >= cols) {
		println(f"i: $i%d rows: $rows%d j: $j%d cols: $cols%d")
		throw new WrongSizeException
	  } else {
		data(i).update(j, v)
	  }
  }

  def apply(i: Int, j: Int): Unit = {
	  if (i >= rows || j >= cols) {
		println(f"i: $i%d rows: $rows%d j: $j%d cols: $cols%d")
		throw new WrongSizeException
	  } else {
		data(i)(j)
	  }
  }

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
	def apply(rows: Int, cols: Int, values: Array[Double]): AAMatrix = {
		require(values.length == rows * cols, "Invalid data length")
		AAMatrix(values.grouped(cols).toArray)
	}
	def zeros(rows: Int, cols: Int): AAMatrix = {
		val init = Array.fill[Array[Double]](rows) 
		                 {Array.fill[Double](cols)(0)}
		
		AAMatrix(init)
	}
}
