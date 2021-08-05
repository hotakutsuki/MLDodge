using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemyController : MonoBehaviour
{
    public Transform playerPos;
    Vector3 direction;
    float resolution = 100;
    //Rigidbody2D rb;

    void Start()
    {
        playerPos = transform.parent.parent.GetChild(0);
        direction = Vector3.Normalize(playerPos.transform.localPosition - transform.localPosition);
        //rb = GetComponent<Rigidbody2D>();
    }

    // Update is called once per frame
    void Update()
    {
        //rb.velocity = Vector3.Normalize(direction) * GameState.speed * Time.deltaTime;

        Vector3 pos = transform.localPosition;
        pos.x = (int)Mathf.Round(pos.x + direction.x * GameState.speed / 30);
        pos.y = (int)Mathf.Round(pos.y + direction.y * GameState.speed / 30);
        transform.localPosition = new Vector3(pos.x, pos.y, pos.z);

        if (Mathf.Abs(transform.localPosition.x) > 5.5f*100 || Mathf.Abs(transform.localPosition.y) > 5.5f*100)
            GameObject.Destroy(gameObject);

    }
}
