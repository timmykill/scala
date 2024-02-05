package tools

object Timings {
	def time[T] (tag: String, a: => T) = {
		System.gc()
		val before = System.currentTimeMillis()
		val result = a
		val after = System.currentTimeMillis()
		println(tag, after - before)
		result
	}
}
