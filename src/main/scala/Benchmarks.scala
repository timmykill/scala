import matrix.{AAMatrix, NewAAMatrix}
import tools.Timings
import scala.util.Random
import scala.math.pow

object Benchmarks {
	def start(args: Array[String]): Unit = {
		val SIZE = 512
		val r = Random
		val a = (0 until pow(SIZE.toDouble, 2).toInt).map(a => r.nextDouble()).toArray
		val b = (0 until pow(SIZE.toDouble, 2).toInt).map(a => r.nextDouble()).toArray
		
		println("im about to start NewAAMatrix multiplication")
		Thread.sleep(2000)
		Timings.time("NewAAMatrix", {
			val Aa = NewAAMatrix(SIZE, a.grouped(SIZE).toArray)
			val Ab = NewAAMatrix(SIZE, b.grouped(SIZE).toArray)
			val Ac = Aa multiply Ab
			//println(Ac.toString)
		})

		println("im about to start AAMatrix multiplication")
		Thread.sleep(2000)
		Timings.time("AAMatrix", {
			val Aa = AAMatrix(a.grouped(SIZE).toArray)
			val Ab = AAMatrix(b.grouped(SIZE).toArray)
			val Ac = Aa multiply Ab
			//println(Ac.toString)
		})

	
  }
}
