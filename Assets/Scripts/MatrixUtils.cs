using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MatrixUtils : MonoBehaviour
{
    /*PRINT*/
    public static void printMatrix(int[] matrix)
    {
        string s = "\n[";
        for (int i = 0; i < matrix.Length; i++)
        {
            s = s + matrix[i] + ", ";
            if (i != matrix.Length - 1)
                s = s + " \n";
        }
        s = s + "]";
        Debug.Log(s);
    }

    public static void printMatrix(double[] matrix)
    {
        string s = "\n[";
        for (int i = 0; i < matrix.Length; i++)
        {
            s = s + matrix[i].ToString("N4") + ", ";
            if (i != matrix.Length - 1)
                s = s + " \n";
        }
        s = s + "]";
        Debug.Log(s);
    }

    public static void printMatrix(float[] matrix)
    {
        string s = "\n[";
        for (int i = 0; i < matrix.Length; i++)
        {
            s = s + matrix[i].ToString("N4") + ", ";
            if (i != matrix.Length - 1)
                s = s + " \n";
        }
        s = s + "]";
        Debug.Log(s);
    }

    public static void printMatrix(int[,] matrix)
    {
        string s = "\n[";
        for (int j = 0; j < matrix.GetLength(0); j++)
        {
            for (int i = 0; i < matrix.GetLength(1); i++)
                s = s + matrix[j, i] + ", ";
            if (j != matrix.GetLength(0) - 1)
                s = s + " \n";
        }
        s = s + "]";
        Debug.Log(s);
    }

    public static string printMatrix(double[,] matrix)
    {
        string s = "\n{{";
        for (int j = 0; j < matrix.GetLength(0); j++)
        {
            for (int i = 0; i < matrix.GetLength(1); i++)
                s = s + matrix[j, i].ToString("N4") + (i == matrix.GetLength(1)-1? "}" : ", ");
            if (j != matrix.GetLength(0) - 1)
                s = s + ",\n{";
        }
        s = s + "}";
        Debug.Log(s);
        return s;
    }

    public static void printMatrix(float[,] matrix)
    {
        string s = "\n[";
        for (int j = 0; j < matrix.GetLength(0); j++)
        {
            for (int i = 0; i < matrix.GetLength(1); i++)
                s = s + matrix[j, i].ToString("N4") + ", ";
            if (j != matrix.GetLength(0) - 1)
                s = s + " \n";
        }
        s = s + "]";
        Debug.Log(s);
    }

    /*MULTIPLY*/
    public static double[,] multiplyMatrix(int[,] a, double[,] b)
    {
        if (a.GetLength(1) != b.GetLength(0)) {
            Debug.Log("Error input matrix");
            throw new System.Exception("Wrong arguments");
        }
            

        double[,] result = new double[a.GetLength(0), b.GetLength(1)];

        for (int j = 0; j < b.GetLength(1); j++)
            for (int k = 0; k < a.GetLength(0); k++)
                for (int i = 0; i < a.GetLength(1); i++)
                    result[k, j] = result[k, j] + a[k, i] * b[i, j];

        return result;
    }

    public static double[,] multiplyMatrix(double[,] a, int[] b)
    {
        int[,] aux = Dto2D(b);
        return multiplyMatrix(a, aux);
    }

    public static double[,] multiplyMatrix(double[,] a, double[] b)
    {
        double[,] aux = Dto2D(b);
        return multiplyMatrix(a, aux);
    }

    public static double[,] multiplyMatrix(double[,] a, int[,] b)
    {
        if (a.GetLength(1) != b.GetLength(0))
            throw new System.Exception("Wrong arguments");

        double[,] result = new double[a.GetLength(0), b.GetLength(1)];

        for (int j = 0; j < b.GetLength(1); j++)
            for (int k = 0; k < a.GetLength(0); k++)
                for (int i = 0; i < a.GetLength(1); i++)
                    result[k, j] = result[k, j] + a[k, i] * b[i, j];

        return result;
    }

    public static double[,] multiplyMatrix(double[,] a, double[,] b)
    {
        if (a.GetLength(1) != b.GetLength(0))
        {
            Debug.Log(a.GetLength(0) + "," + a.GetLength(1) + " X " + b.GetLength(0) + "," + b.GetLength(1));
            throw new System.Exception("Wrong arguments");
        }
            

        double[,] result = new double[a.GetLength(0), b.GetLength(1)];

        for (int j = 0; j < b.GetLength(1); j++)
            for (int k = 0; k < a.GetLength(0); k++)
                for (int i = 0; i < a.GetLength(1); i++)
                    result[k, j] = result[k, j] + a[k, i] * b[i, j];

        return result;
    }

    public static int[,] multiplyMatrix(int[,] a, int[,] b)
    {
        if (a.GetLength(1) != b.GetLength(0))
            throw new System.Exception("Wrong arguments");

        int[,] result = new int[a.GetLength(0), b.GetLength(1)];

        for (int j = 0; j < b.GetLength(1); j++)
            for (int k = 0; k < a.GetLength(0); k++)
                for (int i = 0; i < a.GetLength(1); i++)
                    result[k, j] = result[k, j] + a[k, i] * b[i, j];

        return result;
    }

    public static double[,] multiplyMatrix(double[,] a, float b)
    {
        double[,] result = new double[a.GetLength(0), a.GetLength(1)];

        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
                result[i, j] = a[i, j] * b;

        return result;
    }

    public static double[,] multiplyMatrix(float[,] a, float b)
    {
        double[,] result = new double[a.GetLength(0), a.GetLength(1)];

        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
                result[i, j] = a[i, j] * b;

        return result;
    }

    public static double[] multiplyMatrix(float[] a, float b)
    {
        double[] result = new double[a.GetLength(0)];
        for (int i = 0; i < a.GetLength(0); i++)
            result[i] = a[i] * b;

        return result;
    }

    /*getColumns - getColumns*/
    public static double[] getColumn(double[,] a, int pos)
    {
        if (pos > a.GetLength(1) - 1)
            throw new System.Exception("Wrong arguments");

        double[] result = new double[a.GetLength(0)];
        for (int i = 0; i < a.GetLength(0); i++)
            result[i] = a[i, pos];

        return result;
    }

    public static double[] getRow(double[,] a, int pos)
    {
        if (pos > a.GetLength(0) - 1)
            throw new System.Exception("Wrong arguments");

        double[] result = new double[a.GetLength(1)];
        for (int i = 0; i < a.GetLength(1); i++)
            result[i] = a[pos, i];

        return result;
    }

    /*TRANSP*/
    public static double[,] transportMatrix(double[,] a)
    {
        double[,] result = new double[a.GetLength(1), a.GetLength(0)];

        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
                result[j, i] = a[i, j];

        return result;
    }

    public static double[,] transportMatrix(double[] b)
    {
        double[,] a = Dto2D(b);

        double[,] result = new double[a.GetLength(1), a.GetLength(0)];

        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
                result[j, i] = a[i, j];

        return result;
    }

    public static double[,] transportMatrix(float[] b)
    {
        double[,] a = Dto2D(b);

        double[,] result = new double[a.GetLength(1), a.GetLength(0)];

        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
                result[j, i] = a[i, j];

        return result;
    }

    /*SUM*/
    public static int[,] addMatrix(int[,] a, int[,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            throw new System.Exception("Wrong arguments");

        int[,] result = new int[a.GetLength(0), a.GetLength(1)];

        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
                result[i, j] = a[i, j] + b[i, j];

        return result;
    }

    public static double[,] addMatrix(double[,] a, int[,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            throw new System.Exception("Wrong arguments");

        double[,] result = new double[a.GetLength(0), a.GetLength(1)];

        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
                result[i, j] = a[i, j] + b[i, j];

        return result;
    }

    public static double[,] addMatrix(int[,] a, double[,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            throw new System.Exception("Wrong arguments");

        double[,] result = new double[a.GetLength(0), a.GetLength(1)];

        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
                result[i, j] = a[i, j] + b[i, j];

        return result;
    }

    public static double[,] addMatrix(double[,] a, double[,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
        {
            Debug.Log(a.GetLength(0) + "," + a.GetLength(1) + " + " + b.GetLength(0) + "," + b.GetLength(1));
            throw new System.Exception("Wrong arguments");
        }
            

        double[,] result = new double[a.GetLength(0), a.GetLength(1)];

        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
                result[i, j] = a[i, j] + b[i, j];

        return result;
    }

    public static double[,] addMatrix(double[,] a, double[] b)
    {
        return addMatrix(a, Dto2D(b));
    }

    /*Get largest number or index*/
    public static double getMaxNumber(double[,] a)
    {
        double max = a[0, 0];
        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
            {
                if (a[i, j] < max)
                {
                    max = a[i, j];
                }
            }

        return max;  
    }

    public static int[] getMaxIndex(double[,] a)
    {
        double max = a[0, 0];
        int[] result = new int[2];
        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
            {
                if (a[i, j] > max)
                {
                    max = a[i, j];
                    result[0]= i;
                    result[1]= j;
                }
            }

        return result;
    }

    /**Normilize*/

    public static double[,] normalizeMatrix(double[,] a)
    {
        double max = a[0, 0];
        double min = a[0, 0];
        double[,] result = new double[a.GetLength(0), a.GetLength(1)];

        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
            {
                if (a[i, j] > max)
                    max = a[i, j];
                if (a[i, j] < min)
                    min = a[i, j];
            }

        double m = ((2) / (max - min));
        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
                result[i, j] = m * (a[i, j] - min) - 1 ; 

        return result;
    }

    /*Generate Matrix*/
    public static double[,] generateRandomMatrix(int n, int m)
    {
        double[,] result = new double[n, m];

        for (int j = 0; j < m; j++)
            for (int i = 0; i < n; i++)
                result[i, j] = Random.value;

        return result;
    }

    public static double[,] generateRandomMatrix(int n, int m, float min, float max)
    {
        double[,] result = new double[n, m];

        for (int j = 0; j < m; j++)
            for (int i = 0; i < n; i++)
                result[i, j] = Random.Range(min, max);

        return result;
    }

    public static int[,] generateRandomIntMatrix(int n, int m, int min, int max)
    {
        int[,] result = new int[n, m];

        for (int j = 0; j < m; j++)
            for (int i = 0; i < n; i++)
                result[i, j] = Random.Range(min, max);

        return result;
    }

    /*Mix matrix*/
    internal static double[,] mixRandomly(double[,] a, double[,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
        {
            Debug.Log("Error input matrix");
            throw new System.Exception("Wrong arguments");
        }

        double[,] result = new double[a.GetLength(0), a.GetLength(1)];


        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
            {
                float ran = Random.value;
                if (ran > 0.5)
                    result[i, j] = a[i, j];
                else
                    result[i, j] = b[i, j];
            }
        return result;
    }

    internal static double[,] mixRandomlyWithMutation(double[,] a, double[,] b, float mutationRate, float min, float max)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
        {
            Debug.Log("Error input matrix");
            throw new System.Exception("Wrong arguments");
        }

        double[,] result = new double[a.GetLength(0), a.GetLength(1)];


        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
            {
                float ran = Random.value;
                if (ran < mutationRate)
                    result[i, j] = Random.Range(min, max);
                else
                    if (ran > 0.5 + mutationRate / 2)
                    result[i, j] = a[i, j];
                else
                    result[i, j] = b[i, j];            
            }
        return result;
    }

    /*Sigmoid Function*/
    public static double ReLU (double d)
    {
        if (d > 0)
            return d;
        else
            return 0;
    }

    public static int ReLU(int i)
    {
        if (i > 0)
        {
            return i;
        }
        else
        {
            return 0;
        }
    }

    public static double[] ReLU (double[] a)
    {
        double[] result;
        result = new double[a.GetLength(0)];
        for (int i = 0; i < a.GetLength(0); i++)
            result[i] = ReLU(a[i]);
        return result;
    }

    public static double[,] ReLU(double[,] a)
    {
        double[,] result = new double[a.GetLength(0), a.GetLength(1)];

        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
                result[i, j] = ReLU(a[i, j]);

        return result;
    }

    public static int[] ReLU(int[] a)
    {
        int[] result;
        result = new int[a.GetLength(0)];
        for (int i = 0; i < a.GetLength(0); i++)
            result[i] = ReLU(a[i]);
        return result;
    }

    public static int[,] ReLU(int[,] a)
    {
        int[,] result = new int[a.GetLength(0), a.GetLength(1)];

        for (int j = 0; j < a.GetLength(1); j++)
            for (int i = 0; i < a.GetLength(0); i++)
                result[i, j] = ReLU(a[i, j]);

        return result;
    }


    /*EXTRAS*/
    private static double[] _2Dto1D(double[,] a)
    {
        double[] result;
        if (a.GetLength(0) < a.GetLength(1))
        {
            result = new double[a.GetLength(1)];
            for (int i = 0; i < a.GetLength(1); i++)
                result[i] = a[i, 0];
            return result;
        }
        else
        {
            result = new double[a.GetLength(0)];
            for (int i = 0; i < a.GetLength(0); i++)
                result[i] = a[0, i];
            return result;
        }
    }

    private static int[] _2Dto1D(int[,] a)
    {
        int[] result;
        if (a.GetLength(0) < a.GetLength(1))
        {
            result = new int[a.GetLength(1)];
            for (int i = 0; i < a.GetLength(1); i++)
                result[i] = a[i, 0];
            return result;
        }
        else
        {
            result = new int[a.GetLength(0)];
            for (int i = 0; i < a.GetLength(0); i++)
                result[i] = a[0, i];
            return result;
        }
    }

    private static double[,] Dto2D(double[] b)
    {
        double[,] aux = new double[b.Length, 1];
        for (int i = 0; i < b.Length; i++)
            aux[i, 0] = b[i];
        return aux;
    }

    private static double[,] Dto2D(float[] b)
    {
        double[,] aux = new double[b.Length, 1];
        for (int i = 0; i < b.Length; i++)
            aux[i, 0] = b[i];
        return aux;
    }

    private static int[,] Dto2D(int[] b)
    {
        int[,] aux = new int[b.Length, 1];
        for (int i = 0; i < b.Length; i++)
            aux[i, 0] = b[i];
        return aux;
    }

    public static double[,] _3Dto2D(double[,,] m, int layer)
    {
        if (layer > m.GetLength(2))
            throw new System.Exception("Wrong arguments");

        double[,] result = new double[m.GetLength(0), m.GetLength(1)];

        for (int j = 0; j < m.GetLength(1); j++)
            for (int i = 0; i < m.GetLength(0); i++)
                result[i, j] = m[i, j, layer];

        return result;
    }

    public static double[,] Float2Double2D(float[] b)
    {
        double[,] aux = new double[b.Length, 1];
        for (int i = 0; i < b.Length; i++)
            aux[i, 0] = b[i];
        return aux;
    }

    public static double[,] Int2Double2D(int[] b)
    {
        double[,] aux = new double[b.Length, 1];
        for (int i = 0; i < b.Length; i++)
            aux[i, 0] = b[i];
        return aux;
    }

}
