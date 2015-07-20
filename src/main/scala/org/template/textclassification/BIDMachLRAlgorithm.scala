package org.template.textclassification

import java.io.{InputStreamReader, BufferedReader, ByteArrayInputStream, Serializable}

import BIDMat.{CMat,CSMat,DMat,Dict,FMat,FND,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,HMat,IDict,Image,IMat,LMat,Mat,SMat,SBMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import BIDMach.Learner
import BIDMach.models.{FM,GLM,KMeans,KMeansw,LDA,LDAgibbs,Model,NMF,SFA,RandomForest}
import BIDMach.networks.{DNN}
import BIDMach.datasources.{DataSource,MatDS,FilesDS,SFilesDS}
import BIDMach.mixins.{CosineSim,Perplexity,Top,L1Regularizer,L2Regularizer}
import BIDMach.updaters.{ADAGrad,Batch,BatchNorm,IncMult,IncNorm,Telescoping}
import BIDMach.causal.{IPTW}

import io.prediction.controller.{P2LAlgorithm, Params}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.DataFrame

/**
 * Created by burtn on 15/07/15.
 */

case class BIDMachLRAlgorithmParams (
                               regParam  : Double
                               ) extends Params


class BIDMachLRAlgorithm(
                           val sap: BIDMachLRAlgorithmParams
                           ) extends P2LAlgorithm[PreparedData, NativeLRModel, Query, PredictedResult] {
  // Train your model.
  def train(sc: SparkContext, pd: PreparedData): NativeLRModel = {
    new BIDMachLRModel(sc, pd, sap.regParam)
  }

  // Prediction method for trained model.
  def predict(model: NativeLRModel, query: Query): PredictedResult = {
    model.predict(query.text)
  }

}

  class BIDMachLRModel (
                  sc : SparkContext,
                  pd : PreparedData,
                  regParam : Double
                  ) extends Serializable with NativeLRModel {

    private val labels: Seq[Double] = pd.categoryMap.keys.toSeq

    val data = prepareDataFrame(sc, pd, labels)

    private val lrModels = fitLRModels

    def fitLRModels:Seq[(Double, LREstimate)] = {

      Mat.checkMKL
      Mat.checkCUDA
      if (Mat.hasCUDA > 0) GPUmem

      // 3. Create a logistic regression model for each class.
      val lrModels: Seq[(Double, LREstimate)] = labels.map(
        label => {
          val lab = label.toInt.toString

          val (categories, features) = getDMatsFromData(lab, data)

          val mm: Learner = trainGLM(features, FMat(categories))

          //test(categories, features, mm)
          val modelmat = FMat(mm.modelmat)
          val weightSize = size(modelmat)._2 -1

          val weights = modelmat(1,0 to weightSize)

          val weightArray = (for(i <- 0 to weightSize -1) yield weights(0,i).toDouble).toArray

          // Return (label, feature coefficients, and intercept term.
          (label, LREstimate(weightArray, weights(0,weightSize)))
        }
      )
      lrModels
    }

    def predict(text : String): PredictedResult = {
      predict(text, pd, lrModels)
    }

    def trainGLM(traindata:SMat, traincats: FMat): Learner = {
      //min(traindata, 1, traindata) // the first "traindata" argument is the input, the other is output

      val (mm, mopts) = GLM.learner(traindata, traincats, GLM.logistic)
      mopts.what

      mopts.lrate = 1.0
      mopts.reg1weight = regParam
      mopts.batchSize = 1000
      mopts.npasses = 50
      mopts.autoReset = false
      mopts.addConstFeat = true
      mm.train
      mm
    }

    def getDMatsFromData(lab: String, data:DataFrame): (DMat, SMat) = {
      val features = data.select(lab, "features")
      
      val sparseVectorsWithRowIndices = (for (r <- features) yield (r.getAs[SparseVector]("features"), r.getAs[Double](lab))).zipWithIndex

      val triples = for {
        ((vector, innerLabel), rowIndex) <- sparseVectorsWithRowIndices
        (index, value) <- vector.indices zip vector.values
      }  yield (rowIndex.toString + " " + index.toString + " " + value.toString, innerLabel)

      val catTriples = for {
        ((vector, innerLabel), rowIndex) <- sparseVectorsWithRowIndices
      } yield rowIndex.toString + " " + innerLabel.toString + " " + 1

      val cats = catTriples.collect().mkString("\n")
      val feats = triples.map(x => x._1).collect().mkString("\n")

      val catsMat = loadDMatTxt(toBufferedReader(cats), toBufferedReader(cats))

      val featsMat = loadDMatTxt(toBufferedReader(feats), toBufferedReader(feats))

      (full(cols2sparse(catsMat,true)), cols2sparse(featsMat,true))
    }

    def toBufferedReader(input: String): BufferedReader ={
      val is = new ByteArrayInputStream(input.getBytes());
      new BufferedReader(new InputStreamReader(is));
    }

    //See https://github.com/BIDData/BIDMat/blob/master/src/main/scala/BIDMat/HMat.scala , method loadDMatTxt
    def loadDMatTxt(fin:BufferedReader, din:BufferedReader):DMat = {
      //val fin = new BufferedReader(new InputStreamReader(getInputStream(fname, compressed)))
      var nrows = 0
      var firstline = fin.readLine()
      val parts = firstline.split("[\t ,:]+")
      while (firstline != null && firstline.length > 0) {
        firstline = fin.readLine()
        nrows += 1
      }
      fin.close
      val ncols = parts.length
      val out = DMat.newOrCheckDMat(nrows, ncols, null)
      var irow = 0
      while (irow < nrows) {
        val parts = din.readLine().split("[\t ,:]+")
        var icol = 0
        while (icol < ncols) {
          out.data(irow + icol*out.nrows) = parts(icol).toDouble
          icol += 1
        }
        irow += 1
      }
      din.close
      out
    }

    def test(categories: DMat, features: SMat, mm: Learner): Unit = {
      val testdata = features
      val testcats = categories

      //min(testdata, 1, testdata)

      val predcats = zeros(testcats.nrows, testcats.ncols)

      val (nn, nopts) = GLM.predictor(mm.model, testdata, predcats)

      nopts.addConstFeat = true
      nn.predict

      computeAccuracy(FMat(testcats), predcats)
    }

    def computeAccuracy(testcats: FMat, predcats: FMat): Unit = {
      val lacc = (predcats ∙→ testcats + (1 - predcats) ∙→ (1 - testcats)) / predcats.ncols
      lacc.t
      println(mean(lacc))
    }

}
