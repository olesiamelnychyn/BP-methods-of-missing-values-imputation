package com.company.utils.objects;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

public class MainData {
	private SimpleDataSet train = null;
	private SimpleDataSet impute;
	private int[] columnPredictors;
	private int columnPredicted;
	private DataPoint dp;

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
}
