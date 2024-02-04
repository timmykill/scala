import matrix.{AAMatrix, AMatrix}
import scala.util.Random
import scala.math.pow

object Benchmarks {
	def start(args: Array[String]): Unit = {
		val SIZE = 10
		val r = Random
		val a = (0 until pow(SIZE.toDouble, 2).toInt).map(a => r.nextDouble()).toArray
		val b = (0 until pow(SIZE.toDouble, 2).toInt).map(a => r.nextDouble()).toArray

		println("im about to start AAMatrix multiplication")
		Thread.sleep(2000)
		val AAa = AAMatrix(a.grouped(SIZE).toArray.asInstanceOf[Array[Array[Double]]])
		val AAb = AAMatrix(b.grouped(SIZE).toArray.asInstanceOf[Array[Array[Double]]])

		val AAbefore = System.currentTimeMillis()
		val AAc = AAa multiply AAb
		val AAafter = System.currentTimeMillis()
		println("AAMatrix moltiplication, taken:", AAafter - AAbefore)
		println(AAc.toString)
		System.gc()

		println("im about to start AMatrix multiplication")
		Thread.sleep(2000)
		val Abefore = System.currentTimeMillis()
		val Aa = AMatrix(SIZE, a)
		val Ab = AMatrix(SIZE, b)
		println("created")
		
		val Ac = Aa multiply Ab
		val Aafter = System.currentTimeMillis()
		println("AMatrix moltiplication, taken:", Aafter - Abefore)
		println(Ac.toString)
	
  }
}
