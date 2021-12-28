package com.company.imputationMethods;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoints;

import java.util.ArrayList;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;
import static com.company.utils.calculations.MathCalculations.polyValue;

public class PolynomialCurveFitterApacheMethod extends ImputationMethod {
	private int columnPredictor;
	private int order;
	private WeightedObservedPoints obs;
	private double[] coefficients;

	public PolynomialCurveFitterApacheMethod (int columnPredicted, ArrayList<SimpleDataSet> datasets, int columnPredictor, int order) {
		super(columnPredicted, datasets);
		this.columnPredictor = columnPredictor;
		this.order = order;
	}

	public void preprocessData () {
		obs = new WeightedObservedPoints();
		for (DataPoint dp : trainingCopy.getDataPoints()) {
			obs.add(dp.getNumericalValues().get(columnPredictor), dp.getNumericalValues().get(columnPredicted));
		}
	}

	public void fit () {
		PolynomialCurveFitter fitter = PolynomialCurveFitter.create(order);
		coefficients = fitter.fit(obs.toList());
	}

	public double predict (DataPoint dp) {
		return polyValue(coefficients, dp.getNumericalValues().get(columnPredictor));
	}

	public void print () {
		System.out.println(ANSI_PURPLE_BACKGROUND + "PolynomialCurveFitter (columnPredictor=" + columnPredictor + ")" + ANSI_RESET);
		System.out.print("Coefficients: [");
		for (int i = 0; i < coefficients.length - 1; i++) {
			System.out.print(coefficients[i] + ",");
		}
		System.out.println(coefficients[coefficients.length - 1] + "]");
	}
}
