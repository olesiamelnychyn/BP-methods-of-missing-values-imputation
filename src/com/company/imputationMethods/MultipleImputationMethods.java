package com.company.imputationMethods;

import com.company.utils.DatasetManipulation;
import com.company.utils.objects.MainData;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;

import static com.company.utils.calculations.StatCalculations.*;

public class MultipleImputationMethods {
	private SimpleDataSet datasetMissing;
	private boolean weighted;
	private boolean useMultiDimensionalAsDefault;

	public MultipleImputationMethods (SimpleDataSet datasetMissing, boolean weighted, boolean useMultiDimensionalAsDefault) {
		this.datasetMissing = datasetMissing;
		this.weighted = weighted;
		this.useMultiDimensionalAsDefault = useMultiDimensionalAsDefault;
	}

	public ImputationMethod imputeMultiple (MainData data, Statistics stat) {
		// prepare datasets
		DatasetManipulation.getToBeImputedAndTrainDeepCopiesByClosestDistance(data, datasetMissing, datasetMissing.getDataPoints().indexOf(data.getDp()), 12, weighted);

		// After composing train dataset some predictor columns contain same values, such columns' indexes are excluded from predictors.
		if (data.getColumnPredictors().length == 1) {
			double polyRel = getPolyCorr(data);

			if (useMultiplePolynomialForSinglePredictor(stat, polyRel)) {
				return new MultiplePolynomialRegressionJSATMethod(data, 2);
			}

			return new PolynomialCurveFitterApacheMethod(data, getPolynomialOrderSimple(data, 0.0));
		}

		if (getLinearCorr(data) > stat.getThresholds()[2]) {
			// linear regression is a special way of polynomial regression with 1st degree
			// (it is default therefore we do not pass it)
			return new MultiplePolynomialRegressionJSATMethod(data);
		} else if (getPolyCorr(data) > stat.getThresholds()[3]) {
			return new MultiplePolynomialRegressionJSATMethod(data, 2);
		} else if (useMultiDimensionalAsDefault) {
			return new MultiDimensionalMultipleImputation(data, datasetMissing, stat);
		}
		return new MeanImputationMethod(data);
	}

	private boolean useMultiplePolynomialForSinglePredictor (Statistics stat, double polyRel) {
		return (stat.getKurtosis() > 0 && (polyRel < 0.14 || polyRel > 0.47)) || (stat.getKurtosis() < 0 && polyRel < 0.3 && polyRel > 0.15);
	}
}

