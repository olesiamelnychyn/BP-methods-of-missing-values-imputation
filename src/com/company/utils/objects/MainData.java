package com.company.utils.objects;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import static java.lang.Math.abs;
import static jsat.math.MathTricks.min;

/**
 * Helper object that stores main information needed by the imputation methods
 */
public class MainData {
	private SimpleDataSet train = null;
	private SimpleDataSet impute;
	private int[] columnPredictors;
	private int columnPredicted;
	private DataPoint dp;
	private DataPoint[] toCountStepWith = new DataPoint[]{null, null};

	public MainData (int[] columnPredictors, int columnPredicted, DataPoint dp) {
		this.columnPredictors = columnPredictors;
		this.columnPredicted = columnPredicted;
		this.dp = dp;
	}

	public MainData (int[] columnPredictors, int columnPredicted, DataPoint dp, SimpleDataSet train, SimpleDataSet impute) {
		this.columnPredictors = columnPredictors;
		this.columnPredicted = columnPredicted;
		this.dp = dp;
		this.train = train;
		this.impute = impute;
	}

	public DataPoint getDp () {
		return dp;
	}

	public int getColumnPredicted () {
		return columnPredicted;
	}

	public SimpleDataSet getTrain () {
		return train;
	}

	public void setTrain (SimpleDataSet train) {
		this.train = train;
	}

	public SimpleDataSet getImpute () {
		return impute;
	}

	public void setImpute (SimpleDataSet impute) {
		this.impute = impute;
	}

	public int[] getColumnPredictors () {
		return columnPredictors;
	}

	public void setColumnPredictors (int[] columnPredictors) {
		this.columnPredictors = columnPredictors;
	}

	public DataPoint[] getToCountStepWith () {
		return toCountStepWith;
	}

	public void setToCountStepWith (DataPoint[] toCountStepWith) {
		this.toCountStepWith = toCountStepWith;
	}

	public double[] getStep (double val) {
		if (toCountStepWith[0] == null && toCountStepWith[1] == null) {
			return new double[]{};
		}
		if (toCountStepWith[0] == null) {
			return new double[]{abs(toCountStepWith[1].getNumericalValues().get(columnPredicted) - val)};
		}
		if (toCountStepWith[1] == null) {
			return new double[]{abs(toCountStepWith[0].getNumericalValues().get(columnPredicted) - val)};
		}
		return new double[]{abs(toCountStepWith[1].getNumericalValues().get(columnPredicted) - val), abs(toCountStepWith[0].getNumericalValues().get(columnPredicted) - val)};
	}
}
