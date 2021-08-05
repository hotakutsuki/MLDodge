using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ComputeShaderTest : MonoBehaviour
{
    public ComputeShader computeShader;
    public ComputeShader firstCalculation;
    public ComputeShader MatrixMultiplier;
    public RenderTexture renderTexture;
    public GameObject quad;
    public Material destMat;
    [Range(0f, 1f)]
    public float random;

    public struct TwoNumbers
    {
        public float firstFloat;
        public float secondFloat;
        public float ans;
    }

    public struct MatrixCalculation
    {
        public float[,] a;
        public float[,] b;
        public float[,] ans;
    }

    private TwoNumbers[] data;
    private MatrixCalculation[] matrixData;

    void Start()
    {
        renderTexture = new RenderTexture(256, 256, 24);
        renderTexture.enableRandomWrite = true;
        renderTexture.Create();
        int kernelIndex = computeShader.FindKernel("CSMain");
        computeShader.SetTexture(kernelIndex, "Result", renderTexture);
        computeShader.SetFloat("resolution", renderTexture.width);
        quad.GetComponent<MeshRenderer>().material.SetTexture("_MainTex", renderTexture);

        ////////////////////////////////
        //{
        //    TwoNumbers numbers = new TwoNumbers();
        //    numbers.firstFloat = 5f;
        //    numbers.secondFloat = 5f;
        //    data = new TwoNumbers[1];
        //    data[0] = numbers;

        //    int floatSize = sizeof(float);
        //    int stride = floatSize * 3;
        //    ComputeBuffer buffer = new ComputeBuffer(data.Length, stride);
        //    buffer.SetData(data);
        //    firstCalculation.SetBuffer(kernelIndex, "numbers", buffer);
        //    firstCalculation.Dispatch(kernelIndex, 1, 1, 1);
        //    buffer.GetData(data);
        //    buffer.Dispose();
        //}

        /////////////////////////////////////
        MatrixCalculation matrixCalculation = new MatrixCalculation();
        matrixCalculation.a = new float[2, 2];
        matrixCalculation.a[0, 0] = 1;
        matrixCalculation.a[0, 1] = 2;
        matrixCalculation.a[1, 0] = 3;
        matrixCalculation.a[1, 1] = 4;
        matrixCalculation.b = new float[2, 2];
        matrixCalculation.b[0, 0] = 5;
        matrixCalculation.b[0, 1] = 6;
        matrixCalculation.b[1, 0] = 7;
        matrixCalculation.b[1, 1] = 8;

        matrixData = new MatrixCalculation[1];
        matrixData[0] = matrixCalculation;
        int floatSize = sizeof(float);
        int nDimentionX = 2;
        int nDimentionY = 2;
        int nMatrices = 3;
        int stride = floatSize * nDimentionX * nDimentionY * nMatrices;
        ComputeBuffer buffer = new ComputeBuffer(matrixData.Length, stride);
        buffer.SetData(matrixData);
        firstCalculation.SetBuffer(kernelIndex, "calculation", buffer);
        firstCalculation.Dispatch(kernelIndex, 1, 1, 1);
        buffer.GetData(matrixData);
        buffer.Dispose();
        Debug.Log(matrixData[0].ans);
    }

    // Update is called once per frame
    void Update()
    {
        computeShader.SetFloat("random", random);
        computeShader.Dispatch(0, renderTexture.width, renderTexture.height, 1);
    }
}
