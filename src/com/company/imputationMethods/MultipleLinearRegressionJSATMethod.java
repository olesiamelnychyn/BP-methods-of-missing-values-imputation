package com.company.imputationMethods;

import com.company.utils.DatasetManipulation;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.regression.MultipleLinearRegression;
import jsat.regression.RegressionDataSet;

import java.util.ArrayList;
import java.util.Arrays;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class MultipleLinearRegressionJSATMethod extends ImputationMethod {
	private int[] columnPredictors;
	private int degree = 1;
	private RegressionDataSet regressionDataSet;
	private RegressionDataSet regressionTestDataSet;
	private MultipleLinearRegression multipleLinearRegression;

	public MultipleLinearRegressionJSATMethod (int columnPredicted, ArrayList<SimpleDataSet> datasets, int[] columnPredictors) {
		super(columnPredicted, datasets);
		this.columnPredictors = columnPredictors;
	}

	public MultipleLinearRegressionJSATMethod (int columnPredicted, ArrayList<SimpleDataSet> datasets, int[] columnPredictors, int degree) {
		super(columnPredicted, datasets);
		this.columnPredictors = columnPredictors;
		this.degree = degree;
	}

	public void preprocessData () {
		SimpleDataSet trainingCopy2 = DatasetManipulation.excludeNonPredictors(trainingCopy, columnPredictors, columnPredicted);
		SimpleDataSet toBePredicted2 = DatasetManipulation.excludeNonPredictors(toBePredicted, columnPredictors, columnPredicted);
		int[] predictors = Arrays.copyOf(columnPredictors, columnPredictors.length);
		if (degree > 1) {

			for (int i = 0; i < predictors.length; i++) {
				predictors[i] = i + 1;
			}
			predictors = Arrays.copyOf(predictors, predictors.length * degree);
			for (int i = predictors.length / degree; i < predictors.length; i++) {
				predictors[i] = i + 1;
			}
			trainingCopy2 = DatasetManipulation.addPowerColumns(trainingCopy2, degree, predictors, 0);
			toBePredicted2 = DatasetManipulation.addPowerColumns(toBePredicted2, degree, predictors, 0);
		}
		regressionDataSet = new SimpleDataSet(trainingCopy2.shallowClone().getDataPoints().subList(0, trainingCopy2.getSampleSize())).asRegressionDataSet(0);
		regressionTestDataSet = new SimpleDataSet(toBePredicted2.shallowClone().getDataPoints().subList(0, toBePredicted2.getSampleSize())).asRegressionDataSet(0);
	}

	public void fit () {
		multipleLinearRegression = new MultipleLinearRegression();
		multipleLinearRegression.train(regressionDataSet);
	}

	public double predict (DataPoint dp) {
		DataPoint dataPoint = regressionTestDataSet.getDataPoint(toBePredicted.getDataPoints().indexOf(dp));
		return multipleLinearRegression.regress(dataPoint);
	}

	public void print () {
		System.out.println(ANSI_PURPLE_BACKGROUND + "MultipleLinearRegressionJSAT" + ANSI_RESET);
		System.out.println("Weights: " + multipleLinearRegression.getRawWeight());
	}

}
