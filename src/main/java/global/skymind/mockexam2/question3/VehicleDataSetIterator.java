package global.skymind.mockexam2.question3;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.Random;

public class VehicleDataSetIterator {

    private static int imgHeight;
    private static int imgWidth;
    private static int imgChannels;
    private static int nClass;
    private static int batchsize;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static PathFilter pathFilter = new BalancedPathFilter(new Random(1234), BaseImageLoader.ALLOWED_FORMATS,labelMaker);
    static InputSplit train,test;


    private static ImageTransform imgTransform;

    public VehicleDataSetIterator() {

    }

    public void setup(File file, int height, int width, int channels, int numClass, ImageTransform imageTransform,
                      int batchSize, double trainPerc) {

        imgHeight = height;
        imgWidth = width;
        imgChannels = channels;
        nClass = numClass;
        imgTransform = imageTransform;
        batchsize = batchSize;

        /*
         * Your code here
         */

        train = sample[0];
        test = sample[1];

    }

    private static DataSetIterator makeIterator(InputSplit split,boolean training) {

        ImageRecordReader imgRR = new ImageRecordReader(imgHeight, imgWidth, imgChannels, labelMaker);

        if (training && imgTransform!=null){
            imgRR.initialize(split,imgTransform);
        } else {
            imgRR.initialize(split);
        }

        /*
         * Your code here
         */

        return dataSetIterator;
    }


    public DataSetIterator trainIterator() {
        /*
         * Your code here
         */
    }

    public DataSetIterator testIterator() {
        /*
         * Your code here
         */
    }
}
