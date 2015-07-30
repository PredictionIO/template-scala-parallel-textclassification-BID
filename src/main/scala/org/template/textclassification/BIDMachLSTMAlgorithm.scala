package org.template.textclassification

import BIDMach.networks.LSTMnextWord
import BIDMat.{CMat, CSMat, DMat, Dict, FMat, FND, GMat, GDMat, GIMat, GLMat, GSMat, GSDMat, HMat, IDict, Image, IMat, LMat, Mat, SMat, SBMat, SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import BIDMach.Learner
import BIDMach.models.{FM, GLM, KMeans, KMeansw, LDA, LDAgibbs, Model, NMF, SFA, RandomForest}
import BIDMach.networks.{DNN}
import BIDMach.datasources.{DataSource, MatDS, FilesDS, SFilesDS}
import BIDMach.mixins.{CosineSim, Perplexity, Top, L1Regularizer, L2Regularizer}
import BIDMach.updaters.{ADAGrad, Batch, BatchNorm, IncMult, IncNorm, Telescoping}
import BIDMach.causal.{IPTW}

class BIDMachLSTMAlgorithm {


  def prepareModel {
    //    Mat.checkMKL
    //    Mat.checkCUDA
    //
    //    if (Mat.hasCUDA > 0) GPUmem

    Mat.useMKL = false
    val wlim = 10000

    val a0  = loadIMat("data/train00000.imat.lz4")(0, 0 to 100000)
    val igood = find((a0 < wlim) *@ (a0 >= 0));
    val a = a0(0,igood);

    val (nn, opts) = LSTMnextWord.learner(a)

    opts.npasses = 3
    opts.lrate = 0.3f
    opts.batchSize = 1000
    opts.width = 5;
    opts.height = 1;
    opts.dim = 64;
    opts.kind = 3;
    opts.nvocab = wlim;
    opts.autoReset = false
    opts.bylevel = false;
    opts.debug = 1;
    opts.reg1weight = 0.00001

    val dnn = nn.model.asInstanceOf[LSTMnextWord]
    nn.train

    val lres = nn.results.ncols
    println(lres)
    println(nn.results)
    println(mean(nn.results(?, 0 -> (lres - 1)), 2))

  }

}

object BIDMachLSTMAlgorithm {

  def main(args: Array[String]) {

    val algo = new BIDMachLSTMAlgorithm
    algo.prepareModel
  }
}