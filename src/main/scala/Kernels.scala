import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors

class RbfKernelFunc(gamma_s: Double) extends java.io.Serializable{
    var gamma: Double = gamma_s
    def evaluate(x_1: Vector, x_2: Vector): Double = {
        math.exp(-1 * gamma * math.pow(Vectors.sqdist(x_1, x_2),2))
    }
}