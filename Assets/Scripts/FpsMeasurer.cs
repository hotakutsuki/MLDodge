using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class FpsMeasurer : MonoBehaviour
{
    TextMesh tmp;

    void Start()
    {
        tmp = GetComponent<TextMesh>();
    }
    void Update()
    {
        tmp.text = (1 / Time.deltaTime).ToString("0.00");
    }
}
