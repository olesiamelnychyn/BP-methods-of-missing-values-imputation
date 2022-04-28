package com.company.utils;

import java.text.DecimalFormat;

/**
 * class which contains encodings of colorful print
 */
public class ColorFormatPrint {
	public static String ANSI_RESET = "\u001B[0m";
	public static String ANSI_BLACK = "\u001B[30m";
	public static String ANSI_RED = "\u001B[31m";
	public static String ANSI_GREEN = "\u001B[32m";
	public static String ANSI_YELLOW = "\u001B[33m";
	public static String ANSI_BLUE = "\u001B[34m";
	public static String ANSI_PURPLE = "\u001B[35m";
	public static String ANSI_CYAN = "\u001B[36m";
	public static String ANSI_WHITE = "\u001B[37m";
	public static String ANSI_BOLD_ON = "\033[0;1m";
	public static String ANSI_BOLD_OFF = "\033[0;0m";
	public static final String ANSI_BLACK_BACKGROUND = "\u001B[40m";
	public static final String ANSI_RED_BACKGROUND = "\u001B[41m";
	public static final String ANSI_GREEN_BACKGROUND = "\u001B[42m";
	public static final String ANSI_YELLOW_BACKGROUND = "\u001B[43m";
	public static final String ANSI_BLUE_BACKGROUND = "\u001B[44m";
	public static final String ANSI_PURPLE_BACKGROUND = "\u001B[45m";
	public static final String ANSI_CYAN_BACKGROUND = "\u001B[46m";
	public static final String ANSI_WHITE_BACKGROUND = "\u001B[47m";

	private static DecimalFormat df2 = new DecimalFormat("#.##");

	/**
	 * Format passed value
	 * @param value to be formatted
	 */
	public static String format (double value) {
		if (Double.isNaN(value)) {
			return String.valueOf(Double.NaN);
		}
		return df2.format(value).replace(',', '.');
	}
}
