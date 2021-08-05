using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GridCanvas : MonoBehaviour
{
    public GameObject square;
    public PlayerState playerState;
    List<GameObject> grid;
    List<Vector2> positions = new List<Vector2>();
    void Start()
    {
        grid = new List<GameObject>();
        for (int i = 0; i < playerState.inputsGrid.Length/2; i++)
        {
            positions.Add(calculateInstantiatePosition(i, playerState.inputsGrid.Length / 2));
            grid.Add(Instantiate(square, positions[i], new Quaternion(), transform));
        }
    }

    private Vector2 calculateInstantiatePosition(int i, int totalNumber)
    {
        Vector3 parentPos = transform.parent.parent.position;
        int hLenght = (int)Mathf.Ceil(Mathf.Sqrt(totalNumber));
        float floor = Mathf.Floor(i / hLenght);
        float x = (i - floor * hLenght);
        float y = floor;
        return new Vector2(parentPos.x + 550 + x * 10, parentPos.y - 500 + y * 10);
    }

    // Update is called once per frame
    void Update()
    {
        for (int i = 0; i < grid.Count; i ++)
        {
            Color color;
            if (playerState.inputsGrid[2 * i] == 0)
                color = Color.black;
            else if (playerState.inputsGrid[2 * i + 1] == 1)
                color = Color.red;
            else
                color = Color.green;

            grid[i].GetComponent<SpriteRenderer>().color = color;
            grid[i].transform.position = positions[i];
        }
    }
}
