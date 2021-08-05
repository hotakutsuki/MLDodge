using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class CanvasPresenter : MonoBehaviour
{
    PlayerState ps;

    TextMeshPro[] tmps = new TextMeshPro[16];
    public TextMeshPro tmp;

    void Start()
    {
        ps = transform.parent.GetComponent<PlayerState>();
        for (int i=0; i<ps.rayCastHits.Length; i++)
            tmps[i] = GameObject.Instantiate(tmp, transform);
    }

    void Update()
    {
        for (int i = 0; i < ps.rayCastHits.Length; i++)
        {
            tmps[i].text = ps.inputs[2 * i].ToString("0.00");
            if (ps.inputs[2 * i] > 1 || ps.inputs[2 * i + 1] == 1) tmps[i].color = Color.red;
            else tmps[i].color = Color.black;
            tmps[i].rectTransform.anchoredPosition = ps.rayCastHits[i].point - new Vector2(transform.parent.position.x, transform.parent.position.y);
            Debug.DrawLine(transform.position, ps.rayCastHits[i].point, ps.inputs[2 * i + 1] == 1 ? Color.red : Color.gray);
        }
    }
}
