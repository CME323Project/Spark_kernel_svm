import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD

class RbfKernelFunc(gamma_s: Double) extends java.io.Serializable{
    var gamma: Double = gamma_s
    def evaluate(x_1: Vector, x_2: Vector): Double = {
        math.exp(-1 * gamma * math.pow(Vectors.sqdist(x_1, x_2),2))
    }
}

class KernelSVM(training_data:RDD[LabeledPoint], lambda_s: Double, kernel : String = "rbf", gamma: Double = 1D) extends java.io.Serializable{
    var lambda = lambda_s
    var kernel_func = new RbfKernelFunc(gamma)
    var model = training_data.map(x => (x, 0D))
    var data = training_data
    var s = 1D

    /** Packing algorithm **/
    def train(num_iter: Long, pack_size: Int = 1) {
        /** Initialization */
        var working_data = IndexedRDD(data.zipWithUniqueId().map{case (k,v) => (v,(k, 0D))})
        var norm = 0D
        //var yp = 0D
        //var y = 0D
        var alpha = 0D
        var t = 1
        var i = 0
        var j = 0

        /** Training the model with pack updating */
        while (t <= num_iter) {

            var sample = working_data.takeSample(true, pack_size)
            var yp = sample.map(x => (working_data.map{case (k,v) => (v._1.label * v._2 * kernel_func.evaluate(v._1.features, x._2._1.features))}.reduce((a, b) => a + b)))
            var y = sample.map(x => x._2._1.label)
            var local_set = Map[Long, (LabeledPoint, Double)]()
            var inner_prod = Map[(Int, Int), Double]()

            /** Compute kernel inner product pairs*/
            for (i <- 0 until pack_size) {
                for (j <- i until pack_size) {
                    inner_prod = inner_prod + ((i, j) -> kernel_func.evaluate(sample(i)._2._1.features, sample(j)._2._1.features))
                }
            }

            for (i <- 0 until pack_size) {
                t = t+1
                s = (1 - 1D/(t))*s
                for (j <- (i+1) until (pack_size)) {
                    yp(j) = (1 - 1D/(t))*yp(j)
                }
                if (y(i) * yp(i) < 1) {
                    norm = norm + (2*y(i)) / (lambda * t) * yp(i) + math.pow((y(i)/(lambda*t)), 2)*inner_prod((i,i))
                    alpha = working_data.get(sample(i)._1).get._2
                    local_set = local_set + (sample(i)._1 -> (sample(i)._2._1, alpha + (1/(lambda*t*s))))

                    for (j <- (i+1) to (pack_size-1)) {
                        yp(j) = yp(j) + y(j)/(lambda*t) * inner_prod((i,j))
                    }

                    if (norm > (1/lambda)) {
                        s = s * (1/math.sqrt(lambda*norm))
                        norm = (1/lambda)
                        for (j <- (i+1) to (pack_size-1)) {
                            yp(j) = yp(j) /math.sqrt(lambda*norm)
                        }
                    }

                }
            }
            working_data = working_data.multiput(local_set).cache()
        }
        model = working_data.map{case (k, v) => (v._1, v._2)}.filter{case (k,v) => (v > 0)}
        print (model.count())


    }

    val predict = (data: LabeledPoint) => {
        s * (model.map{case (k,v) => v * k.label * kernel_func.evaluate(data.features, k.features)}.reduce((a, b) => a + b))

    }
    def getAccuracy(data: Array[LabeledPoint]): Double = {
        val N_c = data.map(x => (predict(x) * x.label) ).count(x => x>0)
        val N = data.count(x => true)
        (N_c.toDouble / N)

    }
}
object SimpleApp {
    def main(args: Array[String]) {
        val logFile = "README.md" // Should be some file on your system
        val conf = new SparkConf().setAppName("Simple Application")
        val sc = new SparkContext(conf)

        val data =  MLUtils.loadLibSVMFile(sc, "data/a8a.txt")
        val data_train = data.sample(false, 0.1)

        val svm = new KernelSVM(data_train, 1.0, "rbf", 1.0)
        svm.train(1000,10)


        //val test = MLUtils.loadLibSVMFile(sc, "data/a8at.txt")
        val local_test = data.takeSample(false, 1000)
        println(svm.getAccuracy(local_test))

    }
}



