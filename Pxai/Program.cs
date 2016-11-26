using System;
using System.Linq;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Newtonsoft.Json;
using System.Globalization;

namespace Pxai
{
	class MainClass
	{
		private static double ActivationFunction(double x) { return Math.Max(0, x); }
		private static double ActivationFunctionDerivative(double x) { return x < 0 ? 0 : 1; }

		private static Matrix<Double>[] Weights(string path)
		{
			var layers = JsonConvert.DeserializeObject<double[][][]>(File.ReadAllText(path));

			return layers.Select(layer =>
			{
				var rows = layer.Length;
				var cols = layer.Select(row => row.Length).Distinct().Single();

				var result = DenseMatrix.Create(rows, cols, 0);

				var r = 0;
				foreach (var row in layer)
				{
					var c = 0;
					foreach (var value in row)
					{
						result[r, c++] = value;
					}
					r++;
				}

				return result;
			}).ToArray();
		}

		private static Matrix<Double>[] Biases(string path)
		{
			var layers = JsonConvert.DeserializeObject<double[][]>(File.ReadAllText(path));

			return layers.Select(layer =>
			{
				var rows = layer.Length;
				var cols = 1;

				var result = DenseMatrix.Create(rows, cols, 0);

				var r = 0;
				foreach (var value in layer)
				{
					result[r++, 0] = value;
				}

				return result;
			}).ToArray();
		}

		public static void Main(string[] args)
		{
			var folder = DateTime.Now.ToString("yyyy-MM-dd HH-mm-ss");

			var momentum = 0.9;
			var batchSize = 5.0;
			var learningRate = 0.01;

			var random = new Random();

			var kittyImage = Image.FromFile("Kitty.png");
			var kittyCols = kittyImage.Width;
			var kittyRows = kittyImage.Height;

			var kitty = new Bitmap(kittyImage);
			var kittyData = kitty.LockBits(new Rectangle(0, 0, kitty.Width, kitty.Height), ImageLockMode.ReadWrite, kitty.PixelFormat);
			var kittyBuffer = new byte[kitty.Height * kittyData.Stride];

			kitty.UnlockBits(kittyData); // Ok, we have the stride width now

			var kittyMatrixR = DenseMatrix.Create(kittyRows, kittyCols, (row, col) => kitty.GetPixel(col, row).R / 255.0);
			var kittyMatrixG = DenseMatrix.Create(kittyRows, kittyCols, (row, col) => kitty.GetPixel(col, row).G / 255.0);
			var kittyMatrixB = DenseMatrix.Create(kittyRows, kittyCols, (row, col) => kitty.GetPixel(col, row).B / 255.0);

			var B = Biases("Biases.json");
			var W = Weights("Weights.json");

			var MB = B.Select(x => (Matrix<Double>)DenseMatrix.Create(x.RowCount, x.ColumnCount, 0)).ToArray();
			var MW = W.Select(x => (Matrix<Double>)DenseMatrix.Create(x.RowCount, x.ColumnCount, 0)).ToArray();

			var TY = B.Select(x => (Matrix<Double>)DenseMatrix.Create(x.RowCount, x.ColumnCount, 0)).ToArray();
			var TW = W.Select(x => (Matrix<Double>)DenseMatrix.Create(x.RowCount, x.ColumnCount, 0)).ToArray();

			var DY = B.Select(x => (Matrix<Double>)DenseMatrix.Create(x.RowCount, x.ColumnCount, 0)).ToArray();
			var DB = B.Select(x => (Matrix<Double>)DenseMatrix.Create(x.RowCount, x.ColumnCount, 0)).ToArray();
			var DW = W.Select(x => (Matrix<Double>)DenseMatrix.Create(x.RowCount, x.ColumnCount, 0)).ToArray();

			var R = B.Select(x => (Matrix<Double>)DenseMatrix.Create(x.RowCount, x.ColumnCount, 0)).ToArray();
			var Y = B.Select(x => (Matrix<Double>)DenseMatrix.Create(x.RowCount, x.ColumnCount, 0)).ToArray();
			var X = Enumerable.Range(0, 1).Select(x => (Matrix<Double>)DenseMatrix.Create(2, 1, 0)).Concat(B.Select(x => (Matrix<Double>)DenseMatrix.Create(x.RowCount, x.ColumnCount, 0))).ToArray();

			var O = (Matrix<Double>)DenseMatrix.Create(B.Last().RowCount, 1, 0); // DenseMatrix.OfArray(new Double[3, 1] { { kittyMatrixR[row, col] }, { kittyMatrixG[row, col] }, { kittyMatrixB[row, col] } }) as Matrix<Double>;

			var h = (double)(kittyRows);
			var w = (double)(kittyCols);

			var nfi = CultureInfo.CurrentCulture.NumberFormat.Clone() as NumberFormatInfo;
			nfi.NumberDecimalSeparator = ".";

			for (var cycle = 0; true; cycle++)
			{
				var loss = 0.0;

				for (int batch = 0; batch < batchSize; batch++)
				{
					var row = random.Next(kittyRows);
					var col = random.Next(kittyCols);

					var x = col / (double)h - 0.5;
					var y = row / (double)w - 0.5;

					// Forward propagatiom
					X[0][0, 0] = x;
					X[0][1, 0] = y;

					for (var i = 0; i < W.Length; i++)
					{
						W[i].Multiply(X[i], R[i]);
						R[i] += B[i];
						R[i].Map(ActivationFunction, X[i + 1]);
						R[i].Map(ActivationFunctionDerivative, Y[i]);
					}

					O[0, 0] = kittyMatrixR[row, col];
					O[1, 0] = kittyMatrixG[row, col];
					O[2, 0] = kittyMatrixB[row, col];

					R[R.Length - 1].Subtract(O, DY[DY.Length - 1]);

					var _loss = DY[DY.Length - 1].L2Norm();
					loss += _loss * _loss * 0.5;

					for (var i = W.Length - 1; i > 0; i--)
					{
						DY[i].TransposeAndMultiply(X[i], TW[i]);
						W[i].TransposeThisAndMultiply(DY[i], TY[i - 1]);

						DB[i] += DY[i];
						DW[i] += TW[i];

						Y[i - 1].PointwiseMultiply(TY[i - 1], DY[i - 1]);
					}

					DY[0].TransposeAndMultiply(X[0], TW[0]);
					DW[0] += TW[0];
					DB[0] += DY[0];
				}

				for (int i = 0; i < W.Length; i++)
				{
					MB[i] *= batchSize * momentum / learningRate;
					MB[i] -= DB[i];
					MB[i] *= learningRate / batchSize;

					MW[i] *= batchSize * momentum / learningRate;
					MW[i] -= DW[i];
					MW[i] *= learningRate / batchSize;
				
					B[i] += MB[i];
					W[i] += MW[i];

					DB[i].Clear();
					DW[i].Clear();
				}

				if (cycle % 100 == 0)
				{
					var result = 0.0;

					kittyData = kitty.LockBits(new Rectangle(0, 0, kitty.Width, kitty.Height), ImageLockMode.ReadWrite, kitty.PixelFormat);

					for (var row = 0; row < kittyRows; row++)
					{
						for (int col = 0; col < kittyCols; col++)
						{
							var y = row / w;
							var x = col / h;

							X[0][0, 0] = y;
							X[0][1, 0] = x;

							for (var i = 0; i < W.Length; i++)
							{
								W[i].Multiply(X[i], R[i]);
								R[i] += B[i];
								R[i].Map(ActivationFunction, X[i + 1]);
							}

							kittyBuffer[row * kittyData.Stride + (col * 4) + 0] = (Byte)(255.0 * (X[X.Length - 1][2, 0]));
							kittyBuffer[row * kittyData.Stride + (col * 4) + 1] = (Byte)(255.0 * (X[X.Length - 1][1, 0]));
							kittyBuffer[row * kittyData.Stride + (col * 4) + 2] = (Byte)(255.0 * (X[X.Length - 1][0, 0])); //var output = (Matrix<Double>)DenseMatrix.OfArray(new Double[3, 1] { { kittyMatrixR[row, col] }, { kittyMatrixG[row, col] }, { kittyMatrixB[row, col] } });
							kittyBuffer[row * kittyData.Stride + (col * 4) + 3] = 255;

							O[0, 0] = kittyMatrixR[row, col];
							O[1, 0] = kittyMatrixG[row, col];
							O[2, 0] = kittyMatrixB[row, col];

							R[R.Length - 1].Subtract(O, DY[DY.Length - 1]);

							var error = DY[DY.Length - 1].L2Norm();

							result += error * error * 0.5;
						}
					}

					Marshal.Copy(kittyBuffer, 0, kittyData.Scan0, kittyBuffer.Length);

					Directory.CreateDirectory(folder);

					kitty.UnlockBits(kittyData);
					kitty.Save(Path.Combine(folder, $"Kitty-{ DateTime.Now.Ticks }-{ result }.png"), ImageFormat.Png);

					Console.WriteLine(result);
				}
			}
		}
	}
}