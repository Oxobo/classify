package com.ml;

import com.model.ModelGenerator;
import com.pso.PsoClassifier;
import weka.attributeSelection.*;
import weka.core.Debug;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;


public class Classifier {


        public static final String DATASETPATH = "UCI-Lung-Cancer-DataSet.arff";
        public static final String MODElPATH = "model.bin";

        public static void main(String[] args) throws Exception {

            ModelGenerator mg = new ModelGenerator();

            Instances dataset = mg.loadDataset(DATASETPATH);

            //Missing Values
            ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
            replaceMissingValues.setInputFormat(dataset);
            dataset = Filter.useFilter(dataset, replaceMissingValues);

            //NumericToNominal
            NumericToNominal toNominalFilter = new NumericToNominal();
            toNominalFilter.setInputFormat(dataset);
            dataset = Filter.useFilter(dataset, toNominalFilter);

            AttributeSelection attributeSelection = new AttributeSelection();
            ASEvaluation subsetEval = new CfsSubsetEval();

            ASSearch gSearch = new GreedyStepwise();
            attributeSelection.setEvaluator(subsetEval);
            attributeSelection.setSearch(gSearch);

            dataset.randomize(new Debug.Random(1));// if you comment this line the accuracy of the model will be droped from 96.6% to 80%

            // divide dataset to train dataset 80% and test dataset 20%
            int trainSize = (int) Math.round(dataset.numInstances() * 0.75);
            int testSize = dataset.numInstances() - trainSize;


            //feature reduction
            attributeSelection.SelectAttributes(dataset);
            Instances featureReducedData = attributeSelection.reduceDimensionality(dataset);

            //Normalize dataset
            Filter filter = new Normalize();
            filter.setInputFormat(featureReducedData);
            Instances normalDataset = Filter.useFilter(featureReducedData, filter);

            Instances traindataset = new Instances(normalDataset, 0, trainSize);
            Instances testdataset = new Instances(normalDataset, trainSize, testSize);

            // build classifier with train dataset
            PsoClassifier ann = (PsoClassifier) mg.buildClassifier(traindataset);

            // Evaluate classifier with test dataset
            String evalsummary = mg.evaluateModel(ann, traindataset, testdataset);
            System.out.println("Evaluation: " + evalsummary);
            System.out.println(evalsummary);

            //Save model
            mg.saveModel(ann, MODElPATH);


        }



}
