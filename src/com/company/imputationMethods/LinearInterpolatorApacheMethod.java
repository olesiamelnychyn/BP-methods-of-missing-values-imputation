package com.company.imputationMethods;

import com.company.utils.DatasetManipulation;
import com.company.utils.objects.MainData;
import jsat.classifiers.DataPoint;
import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;

import java.util.Arrays;
import java.util.stream.Collectors;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class LinearInterpolatorApacheMethod extends ImputationMethod {
	private int columnPredictor;
	private boolean increasing;
	private PolynomialSplineFunction polynomialSplineFunction;

	public LinearInterpolatorApacheMethod (MainData data, boolean increasing) {
		super(data);
		this.increasing = increasing;
		this.columnPredictor = data.getColumnPredictors()[0];
	}

	public void preprocessData () {
		// if values are decreasing then reverse dataset
		if (!increasing) {
			trainingCopy = DatasetManipulation.reverseDataset(trainingCopy);
		}
	}

	public void fit () {
		LinearInterpolator linearInterpolator = new LinearInterpolator();
		polynomialSplineFunction = linearInterpolator.interpolate(trainingCopy.getDataMatrix().getColumn(columnPredictor).arrayCopy(), trainingCopy.getDataMatrix().getColumn(columnPredicted).arrayCopy());
	}

	public double predict (DataPoint dp) {
		return polynomialSplineFunction.value(dp.getNumericalValues().get(columnPredictor));
	}

	public void print () {
		System.out.println("\n" + ANSI_PURPLE_BACKGROUND + "PolynomialCurveFitter (columnPredictor=" + columnPredictor + ")" + ANSI_RESET);
		System.out.println("\n\nPiecewise functions:");
		Arrays.stream(polynomialSplineFunction.getPolynomials()).forEach(System.out::println);
		System.out.println(Arrays.stream(polynomialSplineFunction.getKnots())
			.mapToObj(String::valueOf)
			.collect(Collectors.joining(", ", "Knots: [", "]")));
	}
}
