using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerState : MonoBehaviour
{
    public float[] inputs = new float[0]; //defined in the scope
    public int[] inputsGrid = new int[0]; //defined in the scope
    public GameObject enemySpawner;
    public float fitness = 0;
    public bool stillAlive = true;
    bool gridInput;
    private float t225 = Mathf.Sqrt(2) - 1;
    public RaycastHit2D[] rayCastHits = new RaycastHit2D[16];
    float resolution = 10;
    //float maxDistance = Mathf.Sqrt(50); //(9^2 + 9^2)
    private SpriteRenderer sprite;
    private List<Vector3> enemiesPositions;
    private int delta = 100;
    void Start()
    {
        sprite = transform.parent.GetChild(1).GetComponent<SpriteRenderer>();
        gridInput = transform.parent.parent.GetComponent<GameState>().gridInput;
    }
    void Update()
    {
        if (!gridInput)
            calculateInputByRaycast();
        else
            calculateInputByGrid();
    }

    private void calculateInputByGrid()
    {
        enemiesPositions = new List<Vector3>();
        for (int i = 0; i < enemySpawner.transform.childCount; i++)
            enemiesPositions.Add(enemySpawner.transform.GetChild(i).localPosition);

        int n = 0;
        for (int j = -500; j < 500; j += 100)
            for (int i = -500; i < 500; i += 100)            
            {
                bool exist = false;
                foreach (Vector3 pos in enemiesPositions)
                {
                    if (pos.x > i && pos.x <= i + delta && pos.y > j && pos.y <= j + delta)
                    {
                        exist = true;
                        break;
                    }
                }
                bool mySquare = transform.localPosition.x > i && transform.localPosition.x <= i + delta && transform.localPosition.y > j && transform.localPosition.y <= j + delta;

                inputsGrid[n++] = exist || mySquare ? 1 : 0;
                inputsGrid[n++] = exist ? 1 : 0;
            }

        if (stillAlive)
            fitness += 1;
        else
            sprite.color = Color.black;
    }

    private void calculateInputByRaycast()
    {
        rayCastHits[0] = Physics2D.Raycast(transform.position, Vector2.up);
        rayCastHits[1] = Physics2D.Raycast(transform.position, new Vector2(-t225, 1));
        rayCastHits[2] = Physics2D.Raycast(transform.position, new Vector2(-1, 1));
        rayCastHits[3] = Physics2D.Raycast(transform.position, new Vector2(-1, t225));
        rayCastHits[4] = Physics2D.Raycast(transform.position, Vector2.left);
        rayCastHits[5] = Physics2D.Raycast(transform.position, new Vector2(-1, -t225));
        rayCastHits[6] = Physics2D.Raycast(transform.position, new Vector2(-1, -1));
        rayCastHits[7] = Physics2D.Raycast(transform.position, new Vector2(-t225, -1));
        rayCastHits[8] = Physics2D.Raycast(transform.position, Vector2.down);
        rayCastHits[9] = Physics2D.Raycast(transform.position, new Vector2(t225, -1));
        rayCastHits[10] = Physics2D.Raycast(transform.position, new Vector2(1, -1));
        rayCastHits[11] = Physics2D.Raycast(transform.position, new Vector2(1, -t225));
        rayCastHits[12] = Physics2D.Raycast(transform.position, Vector2.right);
        rayCastHits[13] = Physics2D.Raycast(transform.position, new Vector2(1, t225));
        rayCastHits[14] = Physics2D.Raycast(transform.position, new Vector2(1, 1));
        rayCastHits[15] = Physics2D.Raycast(transform.position, new Vector2(t225, 1));
        for (int i = 0; i < rayCastHits.Length; i++)
        {
            //Distance and asintotic function result is cuantificated
            //inputs[2 * i] = 300f / ((int)Mathf.Round(rayCastHits[i].distance - 50));
            inputs[2 * i] = ((rayCastHits[i].distance-50)/900 * 2) - 1;
            if (rayCastHits[i].transform.gameObject.tag == "Enemy")
                inputs[2 * i + 1] = 1;
            else
                inputs[2 * i + 1] = 0;
        }

        if (stillAlive)
            foreach (RaycastHit2D ray in rayCastHits)
            {
                if (ray.transform.gameObject.tag == "Enemy")
                    fitness += 15 * (ray.distance - 50) / 10000;
                else
                    fitness += 1 * (ray.distance - 50) / 10000;
            }
        else
            sprite.color = Color.black;
    }

    void OnTriggerEnter2D(Collider2D col)
    {
        stillAlive = false;
    }

}
