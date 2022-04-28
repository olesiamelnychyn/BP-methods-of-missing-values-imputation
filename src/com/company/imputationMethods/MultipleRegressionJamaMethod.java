package com.company.imputationMethods;

import com.company.imputationMethods.regressions.MultipleLinearRegressionJama;
import com.company.utils.DatasetManipulation;
import com.company.utils.objects.MainData;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class MultipleRegressionJamaMethod extends ImputationMethod {
	private int[] columnPredictors;
	private int degree;
	private double[][] regressionTrainingDataSet;
	private double[][] regressionTestDataSet;
	private MultipleLinearRegressionJama multipleLinearRegression;

	public MultipleRegressionJamaMethod (MainData data) {
		super(data);
		this.columnPredictors = data.getColumnPredictors();
		this.degree = 1;
	}

	public MultipleRegressionJamaMethod (MainData data, int degree) {
		super(data);
		this.columnPredictors = data.getColumnPredictors();
		this.degree = degree;

	}

	public void preprocessData () {
		SimpleDataSet trainingCopy2 = DatasetManipulation.excludeNonPredictors(trainingCopy, columnPredictors, columnPredicted);
		SimpleDataSet toBePredicted2 = DatasetManipulation.excludeNonPredictors(toBePredicted, columnPredictors, columnPredicted);

		int[] predictors = Arrays.copyOf(columnPredictors, columnPredictors.length);
		if (degree > 1) {
			//prepare indexes of predictors
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

		regressionTrainingDataSet = DatasetManipulation.toArray(trainingCopy2, predictors);
		regressionTestDataSet = DatasetManipulation.toArray(toBePredicted2, predictors);
	}

	public void fit () {
		try {
			multipleLinearRegression = new MultipleLinearRegressionJama(regressionTrainingDataSet, trainingCopy.getDataMatrix().getColumn(0).arrayCopy());
		} catch (RuntimeException e) {
			if (degree < 5) {
				System.out.println("Fitting was rerun with a bigger degree due to the exception: " + e.getMessage());
				degree++;
				preprocessData();
				fit();
			} else {
				throw e;
			}
		}
	}

	public double predict (DataPoint dp) {
		double[] dataPoint = regressionTestDataSet[toBePredicted.getDataPoints().indexOf(dp)];
		return multipleLinearRegression.predict(dataPoint);
	}

	public void print () {
		System.out.println(ANSI_PURPLE_BACKGROUND + (degree > 1 ? "MultiplePolynomialRegressionJama" : "MultipleLinearRegressionJama ") + ANSI_RESET);
		System.out.println(IntStream.range(0, columnPredictors.length)
			.mapToDouble(multipleLinearRegression::beta)
			.mapToObj(String::valueOf)
			.collect(Collectors.joining(", ", "Coefficients: [", "]")));
	}
}
