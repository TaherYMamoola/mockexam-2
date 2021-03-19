package global.skymind.question3;

/* ===================================================================
 * We will solve a task of classifying vehicles.
 * The dataset contains 5 classes, each with approximately 48 to 60 images
 * Images are of 100x100 to 240x240 RGB
 * Put the dataset into your resource folder
 *
 * Source: https://www.kaggle.com/rishabkoul1/vechicle-dataset
 * ===================================================================
 * TO-DO
 *
 * 1. Get the file from resource
 * 2. Define image augmentation
 *    (a) Perform random cropping
 *    (b) Perform flipping
 *    (c) Perform rotation
 * 3. Put the defined augmentation into a pipeline
 *    (a) Set all probabilities to 0.3
 *    (b) Set shuffle to false
 * 4. Set up the dataset iterator in VehicleDataSetIterator and VehicleClassifier
 * 5. Return the train iterator and test iterator from the instantiated setup
 * 6. Define configuration for model
 * 7. Define configuration for early stopping
 *    (a) use ROC as score calculator and use AUC as the calculator metric
 *    (b) terminate the training if no score improvement in 2 epochs
 *    (c) terminate the training also if there is an invalid iteration score like NaNs
 * 8. Perform some hyperparameter tuning
 *
 * ====================================================================
 * NOTE: Only make changes at sections specified. Do not shift any of the code.
 *
 * Location of each step is marked using the following annotation:
   /*
    * Your code here
    */

import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;

public class VehicleClassifier {

    //Do not change the parameters here
    private static int height = 100;
    private static int width = 100;
    private static int nClass = 5;
    private static int nChannels = 3;
    private static int seed = 1234;

    //Tunable parameters
    private static int batchSize = 12;
    private static double trainPerc = 0.8;
    private  static double lr = 1e-3;

    public static void main(String[] args) {

        /*
         * Your code here
         */

        /*
         * Your code here
         */

        /*
         * Your code here
         */

        /*
         * Your code here
         */

        /*
         * Your code here
         */

        System.out.println("Number of train examples: " + VehicleDataSetIterator.train.length());
        System.out.println("Number of test examples: " + VehicleDataSetIterator.test.length());

        /*
         * Your code here
         */

        /*
         * Your code here
         */

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConfig, config, trainIter);
        EarlyStoppingResult result = trainer.fit();

        MultiLayerNetwork model = (MultiLayerNetwork) result.getBestModel();

        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(testIter);

        System.out.println("Train stats:\n"+evalTrain);
        System.out.println("Test stats:\n"+evalTest);


    }

}
