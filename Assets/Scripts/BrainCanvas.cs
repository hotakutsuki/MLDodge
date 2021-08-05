using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BrainCanvas : MonoBehaviour
{
    Brain brain;
    public GameObject dot;
    public Material green;
    public Material red;

    public List<GameObject[,]> goList;
    public List<float> deltas;
    public List<double[,]> brainInputs;

    void Start()
    {
        brain = transform.parent.GetComponent<Brain>();
        deltas = new List<float>();
        deltas.Add(-1.5f);
        deltas.Add(-5);
        deltas.Add(-1);
        deltas.Add(-.5f);
        deltas.Add(0);
        deltas.Add(.5f);
        deltas.Add(1);
        deltas.Add(3f);
        deltas.Add(3.5f);
        deltas.Add(4);
        deltas.Add(4.5f);
    }

    // Update is called once per frame
    void Update()
    {
        transform.localPosition = new Vector2(-transform.parent.position.x, -transform.parent.position.y);

        brainInputs = new List<double[,]>();
        brainInputs.Add(brain.i);
        brainInputs.Add(brain.w);
        brainInputs.Add(brain.wi);
        brainInputs.Add(brain.bias);
        brainInputs.Add(brain.wib);
        brainInputs.Add(brain.i2);
        brainInputs.Add(brain.w2);
        brainInputs.Add(brain.wi2);
        brainInputs.Add(brain.bias2);
        brainInputs.Add(brain.wib2);
        brainInputs.Add(brain.output);

        if (goList == null) {
            goList = new List<GameObject[,]>();

            int ix = 0;
            foreach (double[,] brainInput in brainInputs)
            {
                GameObject[,] input = new GameObject[brainInput.GetLength(0), brainInput.GetLength(1)];

                for (int j = 0; j < brainInput.GetLength(1); j++)
                    for (int i = 0; i < brainInput.GetLength(0); i++)
                        input[i, j] = Instantiate(dot, transform.parent.parent.position + new Vector3(deltas[ix] + j / 10f, -5.5f - i / 10f, 0), new Quaternion(), transform);

                goList.Add(input);
                ix++;
            }
        }
            
        int idx = 0;
        foreach (double[,] brainInput in brainInputs)
        {
            for (int j = 0; j < brainInput.GetLength(1); j++)
                for (int i = 0; i < brainInput.GetLength(0); i++) {
                    goList[idx][i, j].GetComponent<MeshRenderer>().material = brainInput[i, j] >= 0 ? green : red;

                    goList[idx][i, j].transform.localScale = new Vector3(.1f, .1f, Mathf.Abs((float)brainInput[i, j])/10);

                    Vector3 position = goList[idx][i, j].transform.position;
                    goList[idx][i, j].transform.position
                        = new Vector3(
                            position.x,
                            position.y, 
                            -(float)brainInput[i, j] / 20);
                }
            idx++;
        }

    }
}

