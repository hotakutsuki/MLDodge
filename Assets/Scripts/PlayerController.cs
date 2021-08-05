using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{

    //Rigidbody2D rb;
    Brain brain;
    float resolution = 100;
    public bool manual;
    
    void Start()
    {
        //rb = GetComponent<Rigidbody2D>();
        brain = GetComponent<Brain>();
    }

    void Update()
    {
        Vector3 pos = transform.localPosition;

        if (!manual)
        {
            if (brain.ans == 1)
            {
                pos.x = (int)Mathf.Round(pos.x);
                pos.y = (int)Mathf.Round(pos.y + GameState.speed / 30);
            }
            else if (brain.ans == 2)
            {
                pos.x = (int)Mathf.Round(pos.x);
                pos.y = (int)Mathf.Round(pos.y - GameState.speed / 30);
            }
            else if (brain.ans == 3)
            {
                pos.x = (int)Mathf.Round(pos.x + GameState.speed / 30);
                pos.y = (int)Mathf.Round(pos.y);
            }
            else if (brain.ans == 4)
            {
                pos.x = (int)Mathf.Round(pos.x - GameState.speed / 30);
                pos.y = (int)Mathf.Round(pos.y);
            }
        } else
        {
            if (Input.GetAxis("Vertical") > 0 )
            {
                pos.x = (int)Mathf.Round(pos.x);
                pos.y = (int)Mathf.Round(pos.y + GameState.speed / 30);
            }
            else if (Input.GetAxis("Vertical") < 0 )
            {
                pos.x = (int)Mathf.Round(pos.x);
                pos.y = (int)Mathf.Round(pos.y - GameState.speed / 30);
            }
            else if (Input.GetAxis("Horizontal") > 0 )
            {
                pos.x = (int)Mathf.Round(pos.x + GameState.speed / 30);
                pos.y = (int)Mathf.Round(pos.y);
            }
            else if (Input.GetAxis("Horizontal") < 0 )
            {
                pos.x = (int)Mathf.Round(pos.x - GameState.speed / 30);
                pos.y = (int)Mathf.Round(pos.y);
            }
        }


        if (Mathf.Abs(transform.localPosition.x) > 4.6f*100 || Mathf.Abs(transform.localPosition.y) > 4.6f*100)
            transform.localPosition = new Vector2(4.6f*100, 4.6f*100);
        else
            transform.localPosition = pos;

        //rb.velocity = Vector2.zero;
    }
}
