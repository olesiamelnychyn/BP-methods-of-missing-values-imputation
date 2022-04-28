package com.company.imputationMethods;

import com.company.utils.objects.MainData;
import jsat.classifiers.DataPoint;
import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoints;

import java.util.Arrays;
import java.util.stream.Collectors;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;
import static com.company.utils.calculations.MathCalculations.polyValue;

public class PolynomialCurveFitterApacheMethod extends ImputationMethod {
	private int columnPredictor;
	private int order;
	private WeightedObservedPoints obs;
	private double[] coefficients;

	public PolynomialCurveFitterApacheMethod (MainData data, int order) {
		super(data);
		this.columnPredictor = data.getColumnPredictors()[0];
		this.order = order;
	}

	public PolynomialCurveFitterApacheMethod (MainData data, int order, int indexPredictor) {
		super(data);
		this.columnPredictor = data.getColumnPredictors()[indexPredictor];
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
		System.out.println(Arrays.stream(coefficients)
			.mapToObj(String::valueOf)
			.collect(Collectors.joining(", ", "Coefficients: [", "]")));
	}
}
