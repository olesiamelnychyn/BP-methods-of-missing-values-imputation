package com.company.imputationMethods;

import com.company.utils.DatasetManipulation;
import com.company.utils.objects.MainData;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;

import static com.company.utils.calculations.StatCalculations.*;

public class SimpleImputationMethods {
	SimpleDataSet datasetMissing;

	public SimpleImputationMethods (SimpleDataSet datasetMissing) {
		this.datasetMissing = datasetMissing;
	}

	public ImputationMethod imputeSimple (MainData data, Statistics stat) {
		if (data.getTrain() == null) {
			// prepare datasets
			DatasetManipulation.getToBeImputedAndTrainDeepCopiesAroundIndex(data, datasetMissing, datasetMissing.getDataPoints().indexOf(data.getDp()), 14);
		}

		// select appropriate imputation method
		if (getMeanDevPercent(data) <= stat.getThresholds()[0]) {
			return new MeanImputationMethod(data);
		} else if (getMedianDevPercent(data) <= stat.getThresholds()[1]) {
			return new MedianImputationMethod(data);
		} else if (data.getColumnPredictors().length == 1) {
			int columnPredictor = data.getColumnPredictors()[0];
			if (isStrictlyIncreasing(data, data.getColumnPredicted()) && isStrictlyIncreasing(data, columnPredictor)) {
				return new LinearInterpolatorApacheMethod(data, true);
			} else if (isStrictlyDecreasing(data, data.getColumnPredicted()) && isStrictlyDecreasing(data, columnPredictor)) {
				return new LinearInterpolatorApacheMethod(data, false);
			} else if (getLinearCorr(data) > stat.getThresholds()[2]) {
				return new LinearRegressionJSATMethod(data);
			} else {
				int order = getPolynomialOrderSimple(data, stat.getThresholds()[3]);
				if (order != -1) {
					return new PolynomialCurveFitterApacheMethod(data, order);
				}
			}
			if (useClosest(data)) {
				return new ClosestImputation(data);
			}
		}

		return new MeanImputationMethod(data);
	}

	private boolean useClosest (MainData data) {
		return isWithinInterval(getMeanDevPercent(data), new double[]{0.32, 0.35})
			&& isWithinInterval(getLinearCorr(data), new double[]{0.09, 0.16});
	}
}
