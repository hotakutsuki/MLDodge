using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemySpawner : MonoBehaviour
{
    public GameObject Enemy;
    public GameState gameState;
    float timeToSpawn;
    float presentTime;

    public void SpawnEnemy(float x, float y)
    {
        GameObject.Instantiate(Enemy, transform.position + new Vector3(x, y, transform.localPosition.z), new Quaternion(), transform);
    }
}
