import org.apache.spark.rdd._

import org.apache.spark.mllib.regression.LabeledPoint
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD

class KernelSVM(training_data:RDD[LabeledPoint], lambda_s: Double, kernel : String = "rbf", gamma: Double = 1D) extends java.io.Serializable{
    var lambda = lambda_s
    var kernel_func = new RbfKernelFunc(gamma)
    var model = training_data.map(x => (x, 0D))
    var data = training_data
    var s = 1D

    def train(num_iter: Long, pack_size: Long = 1) {
        /** Initialization */
        var working_data = IndexedRDD(data.zipWithUniqueId().map{case (k,v) => (v,(k, 0D))})
        var norm = 0D
        var yp = 0D
        var y = 0D
        var alpha = 0D
        var t = 1

        /** Training the model with pack updating */
        while (t <= num_iter) {
            var sample = (working_data.takeSample(true, 1))(0)
            y = sample._2._1.label
            yp = working_data.map{case (k,v) => (v._1.label * v._2 * kernel_func.evaluate(v._1.features, sample._2._1.features))}.reduce((a, b) => a + b)
            s = (1 - 1D/(t+1))*s
            if (y * yp < 1) {
                norm = norm + (2*y) / (lambda * t) * yp + math.pow((y/(lambda*t)), 2)*kernel_func.evaluate(sample._2._1.features, sample._2._1.features)
                alpha = working_data.get(sample._1).get._2
                working_data = working_data.put(sample._1, (sample._2._1, alpha + (1/(lambda*t*s)))).cache()
                
                if (norm > (1/lambda)) {
                    s = s * (1/math.sqrt(lambda*norm))
                    norm = (1/lambda)
                }

            }
            t = t+1
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