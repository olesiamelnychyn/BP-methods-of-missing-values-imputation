package com.company.imputationMethods;

import com.company.utils.DatasetManipulation;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;

import java.util.ArrayList;
import java.util.Arrays;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class LinearInterpolatorApacheMethod extends ImputationMethod {
	private int columnPredictor;
	private boolean increasing;
	private PolynomialSplineFunction polynomialSplineFunction;

	public LinearInterpolatorApacheMethod (int columnPredicted, ArrayList<SimpleDataSet> datasets, int columnPredictor, boolean increasing) {
		super(columnPredicted, datasets);
		this.increasing = increasing;
		this.columnPredictor = columnPredictor;
	}

	public void preprocessData () {
		// if values are decreasing than reverse dataset
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
//			double[] knots = polynomialSplineFunction.getKnots();
//			System.out.print("Knots: [");
//			for (int i = 0; i < knots.length - 1; i++) {
//				System.out.print(knots[i] + ",");
//			}
//			System.out.println(knots[knots.length - 1] + "]");
	}
}