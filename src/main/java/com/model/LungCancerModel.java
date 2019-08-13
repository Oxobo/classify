package com.model;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This is a classifier for iris.2D.arff dataset
 * @author Taha Emara
 * Website: http://www.emaraic.com
 * Email  : taha@emaraic.com
 * Created on: Jul 1, 2017
 * Github link: https://github.com/emara-geek/weka-example
 */
public class LungCancerModel {

    private Attribute attribute01;
    private Attribute attribute02;
    private Attribute attribute03;
    private Attribute attribute04;
    private Attribute attribute05;
    private Attribute attribute06;
    private Attribute attribute07;
    private Attribute attribute08;
    private Attribute attribute09;
    private Attribute attribute10;
    private Attribute attribute11;
    private Attribute attribute12;
    private Attribute attribute13;
    private Attribute attribute14;
    private Attribute attribute15;
    private Attribute attribute16;
    private Attribute attribute17;
    private Attribute attribute18;
    private Attribute attribute19;
    private Attribute attribute20;
    private Attribute attribute21;
    private Attribute attribute22;
    private Attribute attribute23;
    private Attribute attribute24;
    private Attribute attribute25;
    private Attribute attribute26;
    private Attribute attribute27;
    private Attribute attribute28;
    private Attribute attribute29;
    private Attribute attribute30;
    private Attribute attribute31;
    private Attribute attribute32;
    private Attribute attribute33;
    private Attribute attribute34;
    private Attribute attribute35;
    private Attribute attribute36;
    private Attribute attribute37;
    private Attribute attribute38;
    private Attribute attribute39;
    private Attribute attribute40;
    private Attribute attribute41;
    private Attribute attribute42;
    private Attribute attribute43;
    private Attribute attribute44;
    private Attribute attribute45;
    private Attribute attribute46;
    private Attribute attribute47;
    private Attribute attribute48;
    private Attribute attribute49;
    private Attribute attribute50;
    private Attribute attribute51;
    private Attribute attribute52;
    private Attribute attribute53;
    private Attribute attribute54;
    private Attribute attribute55;
    private Attribute attribute56;

    private ArrayList<Attribute> attributes;
    private ArrayList<String> classVal;
    private Instances dataRaw;


    public LungCancerModel() {
        attribute01 = new Attribute("a01");
        attribute02 = new Attribute("a02");
        attribute03 = new Attribute("a03");
        attribute04 = new Attribute("a04");
        attribute05 = new Attribute("a05");
        attribute06 = new Attribute("a06");
        attribute07 = new Attribute("a07");
        attribute08 = new Attribute("a08");
        attribute09 = new Attribute("a09");
        attribute10 = new Attribute("a10");
        attribute11 = new Attribute("a11");
        attribute12 = new Attribute("a12");
        attribute13 = new Attribute("a13");
        attribute14 = new Attribute("a14");
        attribute15 = new Attribute("a15");
        attribute16 = new Attribute("a16");
        attribute17 = new Attribute("a17");
        attribute18 = new Attribute("a18");
        attribute19 = new Attribute("a19");
        attribute20 = new Attribute("a20");
        attribute21 = new Attribute("a21");
        attribute22 = new Attribute("a22");
        attribute23 = new Attribute("a23");
        attribute24 = new Attribute("a24");
        attribute25 = new Attribute("a25");
        attribute26 = new Attribute("a26");
        attribute27 = new Attribute("a27");
        attribute28 = new Attribute("a28");
        attribute29 = new Attribute("a29");
        attribute30 = new Attribute("a30");
        attribute31 = new Attribute("a31");
        attribute32 = new Attribute("a32");
        attribute33 = new Attribute("a33");
        attribute34 = new Attribute("a34");
        attribute35 = new Attribute("a35");
        attribute36 = new Attribute("a36");
        attribute37 = new Attribute("a37");
        attribute38 = new Attribute("a38");
        attribute39 = new Attribute("a39");
        attribute40 = new Attribute("a40");
        attribute41 = new Attribute("a41");
        attribute42 = new Attribute("a42");
        attribute43 = new Attribute("a43");
        attribute44 = new Attribute("a44");
        attribute45 = new Attribute("a45");
        attribute46 = new Attribute("a46");
        attribute47 = new Attribute("a47");
        attribute48 = new Attribute("a48");
        attribute49 = new Attribute("a49");
        attribute50 = new Attribute("a50");
        attribute51 = new Attribute("a51");
        attribute52 = new Attribute("a52");
        attribute53 = new Attribute("a53");
        attribute54 = new Attribute("a54");
        attribute55 = new Attribute("a55");
        attribute56 = new Attribute("a56");

        attributes = new ArrayList<>();
        classVal = new ArrayList<>();
        classVal.add("Type1-Lung-Cancer");
        classVal.add("Type2-Lung-Cancer");
        classVal.add("Type3-Lung-Cancer");

        attributes.add(attribute01);
        attributes.add(attribute02);
        attributes.add(attribute03);
        attributes.add(attribute04);
        attributes.add(attribute05);
        attributes.add(attribute06);
        attributes.add(attribute07);
        attributes.add(attribute08);
        attributes.add(attribute09);
        attributes.add(attribute10);
        attributes.add(attribute11);
        attributes.add(attribute12);
        attributes.add(attribute13);
        attributes.add(attribute14);
        attributes.add(attribute15);
        attributes.add(attribute16);
        attributes.add(attribute17);
        attributes.add(attribute18);
        attributes.add(attribute19);
        attributes.add(attribute20);
        attributes.add(attribute21);
        attributes.add(attribute22);
        attributes.add(attribute23);
        attributes.add(attribute24);
        attributes.add(attribute25);
        attributes.add(attribute26);
        attributes.add(attribute27);
        attributes.add(attribute28);
        attributes.add(attribute29);
        attributes.add(attribute30);
        attributes.add(attribute31);
        attributes.add(attribute32);
        attributes.add(attribute33);
        attributes.add(attribute34);
        attributes.add(attribute35);
        attributes.add(attribute36);
        attributes.add(attribute37);
        attributes.add(attribute38);
        attributes.add(attribute39);
        attributes.add(attribute40);
        attributes.add(attribute41);
        attributes.add(attribute42);
        attributes.add(attribute43);
        attributes.add(attribute44);
        attributes.add(attribute45);
        attributes.add(attribute46);
        attributes.add(attribute47);
        attributes.add(attribute48);
        attributes.add(attribute49);
        attributes.add(attribute50);
        attributes.add(attribute51);
        attributes.add(attribute52);
        attributes.add(attribute53);
        attributes.add(attribute54);
        attributes.add(attribute55);
        attributes.add(attribute56);

        attributes.add(new Attribute("cancerType", classVal));
        dataRaw = new Instances("TestInstances", attributes, 0);
        dataRaw.setClassIndex(dataRaw.numAttributes() - 1);
    }


    public String classifiy(Instances insts, String path) {
        String result = "Not classified!!";
        Classifier cls = null;
        try {
            cls = (MultilayerPerceptron) SerializationHelper.read(path);
            result = classVal.get((int) cls.classifyInstance(insts.firstInstance()));
        } catch (Exception ex) {
            Logger.getLogger(LungCancerModel.class.getName()).log(Level.SEVERE, null, ex);
        }
        return result;
    }


    public Instances getInstance() {
        return dataRaw;
    }


}
