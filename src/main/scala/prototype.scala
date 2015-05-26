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

class RbfKernelFunc(sigma_s: Double) extends java.io.Serializable{
    var sigma: Double = sigma_s
    def evaluate(x_1: Vector, x_2: Vector): Double = {
        math.exp(-1 * math.pow(Vectors.sqdist(x_1, x_2),2)/(2 * sigma * sigma))
    }
}

class KernelSVM(training_data:RDD[LabeledPoint], lambda_s:Double, kernel : String, kernel_param: Double, num_iter: Long) extends java.io.Serializable{
    //var sc = sparkContext
    var lambda = lambda_s
    var kernel_func = new RbfKernelFunc(kernel_param)
    //Initialize model as a RDD[(LabeledPoint, Double)]
    var model = training_data.map(x => (x, 0D))
    var data = training_data
    var s = 1D

    def train() {
        var working_data = IndexedRDD(data.zipWithUniqueId().map{case (k,v) => (v,(k, 0D))})
        var norm = 0D
        var yp = 0D
        var y = 0D
        var alpha = 0D
        var t = 1
        while (t <= num_iter) {
            var sample = (working_data.takeSample(true, 1))(0)
            y = sample._2._1.label
            yp = working_data.map{case (k,v) => (v._1.label * v._2 * kernel_func.evaluate(v._1.features, sample._2._1.features))}.reduce((a, b) => a + b)
            s = (1 - 1D/(t+1))*s
            if (y * yp < 1) {
                norm = norm + (2*y) / (lambda * t) * yp + math.pow((y/(lambda*t)), 2)*kernel_func.evaluate(sample._2._1.features, sample._2._1.features)
                alpha = working_data.get(sample._1).get._2
                working_data = working_data.put(sample._1, (sample._2._1, alpha + (1/(lambda*t*s))))
                
                if (norm > (1/lambda)) {
                    s = s * (1/math.sqrt(lambda*norm))
                    norm = (1/lambda)
                }

            }
            t = t+1
        }
        model = working_data.map{case (k, v) => (v._1, v._2)}.filter{case (k,v) => (v > 0)}
        print (model.count())
        println("fuck")


    }

    val predict = (data: LabeledPoint) => {
        s * (model.map{case (k,v) => v * k.label * kernel_func.evaluate(data.features, k.features)}.reduce((a, b) => a + b))

    }
    def getAccuracy(data: Array[LabeledPoint]): Double = {
        val N_c = data.map(x => (predict(x) * x.label) ).count(x => x>0)
        val N = data.count(x => true)
        println(N_c)
        println(N)
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

        val svm = new KernelSVM(data_train, 1.0, "rbf", 1.0, 100)
        svm.train()


        //val test = MLUtils.loadLibSVMFile(sc, "data/a8at.txt")
        val local_test = data.takeSample(false, 1000)
        println(svm.getAccuracy(local_test))

    }
}



