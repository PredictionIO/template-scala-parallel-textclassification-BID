package org.template.textclassification


import io.prediction.controller.PPreparator
import io.prediction.controller.Params
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.{IDF, IDFModel, HashingTF}
import org.apache.spark.mllib.linalg.{Matrix, DenseMatrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.immutable.HashMap
import scala.collection.JavaConversions._
import scala.math._


// 1. Initialize Preparator parameters. Recall that for our data
// representation we are only required to input the n-gram window
// components.

case class PreparatorParams(
                             nGram: Int
                             ) extends Params


// 2. Initialize your Preparator class.

class Preparator(pp: PreparatorParams) extends PPreparator[TrainingData, PreparedData] {

  // Prepare your training data.
  def prepare(sc: SparkContext, td: TrainingData): PreparedData = {
    new PreparedData(td, pp.nGram, sc)
  }
}

//------PreparedData------------------------

class PreparedData(
                    val td: TrainingData,
                    val nGram: Int,
                    @transient val sc: SparkContext
                    ) extends Serializable {


  // 1. Hashing function: Text -> term frequency vector.

  private val hasher = new HashingTF(500)

  private def hashTF(text: String): Vector = {
    val newList: Array[String] = text.split(" ")
      .sliding(nGram)
      .map(_.mkString)
      .toArray

    hasher.transform(newList)
  }


  private def calculateSPPMI(localMat: Matrix, N: Long, k: Int): IndexedSeq[MatrixEntry] = {

    println(localMat)
    val pmiMatrixEntries = for (i <- 0 until localMat.numCols; j <- 0 until localMat.numRows)
      yield {
        new MatrixEntry(j, i, math.max(0, math.log(localMat(j, i) * N / (localMat(i, i) * localMat(j, j))) / math.log(2.0) - math.log(k) / math.log(2.0)))
      }
    return pmiMatrixEntries
  }

  private def generateSPPMIMatrix(trainData: TrainingData, sc:SparkContext) : Map[String,Vector] = {
    val hashedFeats = trainData.data.map(e => hashTF(e.text))
    val indexedRows = hashedFeats.zipWithIndex.map(x => new IndexedRow(x._2, x._1))


    val blockMat: BlockMatrix = new IndexedRowMatrix(indexedRows).toBlockMatrix

    //println(blockMat.numCols())
    //println(blockMat.numRows())
    val cooccurrences = blockMat.transpose.multiply(blockMat)
    val locMat = cooccurrences.toLocalMatrix.asInstanceOf[DenseMatrix].toSparse
    val k = 1

    val pmiEntries = calculateSPPMI(locMat, blockMat.numRows, k)

    val pmiMat: CoordinateMatrix = new CoordinateMatrix(sc.parallelize(pmiEntries))

    val indexedPMIMat = pmiMat.toIndexedRowMatrix()

    //val svdedPMImat = indexedPMIMat.computeSVD(50).U


    println(trainData.data.count())
    println(indexedPMIMat.rows.count())
    //println(svdedPMImat.rows.count())

    val pmiMatRows = indexedPMIMat.rows.map(e=> e.index -> e.vector).collectAsMap()

    //TODO: take into account feature counts, currently it's on/off
    //also not use var
    //j<- 0 until v(i).toIn
    val composedWordVectors = for(v<- hashedFeats)
      yield {
        var ar =  Array.fill[Double](pmiMatRows.head._2.size)(0)
        for( i <- 0 until v.size; if v(i) > 0){
          ar = (ar,pmiMatRows(i).toArray).zipped.map(_ + _)
          //ar = ar ++ pmiMatRows(i).toArray
        }
        Vectors.dense(ar.map(x=> x/v.size)).toSparse }

    val textToSPPMIVectorMap = (trainData.data.map(x=> x.text) zip composedWordVectors).collect.toMap

    return textToSPPMIVectorMap
  }

  val ppmiMap = generateSPPMIMatrix(td,sc)
  println(ppmiMap.head._2.size)

  // 2. Term frequency vector -> t.f.-i.d.f. vector.

  //val idf : IDFModel = new IDF().fit(td.data.map(e => hashTF(e.text)))


  // 3. Document Transformer: text => tf-idf vector.

  //  def transform(text : String): Vector = {
  //    // Map(n-gram -> document tf)
  //    idf.transform(hashTF(text))
  //  }



   def transform(text : String): Vector = {
      // Map(n-gram -> document tf)

      val result = ppmiMap(text)
      //println(result)
      result
    }




  // 4. Data Transformer: RDD[documents] => RDD[LabeledPoints]

  val transformedData: RDD[(LabeledPoint)] = {
    td.data.map(e => LabeledPoint(e.label, transform(e.text)))
  }


  // 5. Finally extract category map, associating label to category.
  val categoryMap = td.data.map(e => (e.label, e.category)).collectAsMap


}




