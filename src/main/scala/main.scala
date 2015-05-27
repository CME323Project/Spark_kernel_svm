import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._

import org.apache.spark.mllib.util.MLUtils

object TestKernelSVM {
    def main(args: Array[String]) {
        val logFile = "README.md" // Should be some file on your system
        val conf = new SparkConf().setAppName("KernelSVM Test")
        val sc = new SparkContext(conf)

        val data =  MLUtils.loadLibSVMFile(sc, "data/a8a.txt")

        val data_train = data.sample(false, 0.8)

        val svm = new KernelSVM(data_train, 1.0, "rbf", 1.0)
        svm.train(1000,10)

        //val test = MLUtils.loadLibSVMFile(sc, "data/a8at.txt")
        val local_test = data.takeSample(false, 1000)
        println(svm.getAccuracy(local_test))

        println(svm.getNumSupportVectors())

    }
}