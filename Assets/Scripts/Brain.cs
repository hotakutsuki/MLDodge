using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class Brain : MonoBehaviour
{
    PlayerState ps;
    
    //First layer
    public double[,] i;
    public double[,] w;
    public double[,] wi;
    public double[,] bias;
    public double[,] wib;

    //Second layer
    public double[,] i2;
    public double[,] w2;
    public double[,] wi2;
    public double[,] bias2;
    public double[,] wib2;

    //Third layer
    public double[,] i3;
    public double[,] w3;
    public double[,] wi3;
    public double[,] bias3;
    public double[,] wib3;

    //Output
    public double[,] output;

    public int ans;
    void Start()
    {
        ps = gameObject.GetComponent<PlayerState>();
    }

    // Update is called once per frame
    void Update()
    {
        if (w!=null && bias!= null && w2 != null && bias2 != null)
        {
            if (transform.parent.parent.GetComponent<GameState>().gridInput)
                i = MatrixUtils.Int2Double2D(ps.inputsGrid);
            else
                i = MatrixUtils.Float2Double2D(ps.inputs);
            wi = MatrixUtils.multiplyMatrix(w, i);
            wib = MatrixUtils.addMatrix(wi, bias);

            i2 = MatrixUtils.ReLU(wib);
            wi2 = MatrixUtils.multiplyMatrix(w2, i2);
            wib2 = MatrixUtils.addMatrix(wi2, bias2);
            if (GameState.thirLayers)
            {
                i3 = MatrixUtils.ReLU(wib2);
                wi3 = MatrixUtils.multiplyMatrix(w3, i3);
                wib3 = MatrixUtils.addMatrix(wi3, bias3);

                output = MatrixUtils.ReLU(wib3);
            } else
            {
                output = MatrixUtils.ReLU(wib2);
            }
            
            int[] result = MatrixUtils.getMaxIndex(output);
            ans = result[0];
        }
    }
}
