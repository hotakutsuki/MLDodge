using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameState : MonoBehaviour
{
    public static float speed = 7*100;
    public bool gridInput;
    public static bool thirLayers = true;
    public int poolSize;
    private int generation = 1;
    public static int timeOfPlay = 55;
    public float mutationRate;
    public bool endless = true;
    public GameObject gameInstance;
    private List<GameObject> gameInstances = new List<GameObject>();
    List<Brain> aprovedBrainsPool = new List<Brain>();
    List<Brain> allBrainsPool = new List<Brain>();

    List<Brain> auxBrainsPool = new List<Brain>();
    List<float> fitneses = new List<float>();

    public static ArrayList w1GenPool = new ArrayList();
    public static ArrayList bias1GenPool = new ArrayList();
    public static ArrayList w2GenPool = new ArrayList();
    public static ArrayList bias2GenPool = new ArrayList();

    private int gamePhases;
    private int gamePhase;
    private int numberOfGames;

    public bool random;
    public bool circle;

    public bool blockUp;
    public bool blockDown;
    public bool blockLeft;
    public bool blockRight;

    float fitnessArg;

    public float timeToSpawn;
    float presentTime;

    List<Vector3> positions;
    int presentPosition;
    public int enemiesStartPositionInCircle;

    private bool manual;
    void Start()
    {
        Application.targetFrameRate = 300;
        if (transform.childCount > 0)
            manual = transform.GetChild(0).GetChild(0).GetComponent<PlayerController>().manual;
        //Debug.Log(HandleTextFile.ReadString());
        gamePhases = (int)Mathf.Floor((poolSize-1) / 1000);
        gamePhase = 0;
        presentPosition = enemiesStartPositionInCircle;
        //loadSmartBrains();
        generateRandomPool();
        createPostions();
        startPhaseGame();
        /*this is for the enemy spawner*/
        //timeToSpawn = enemyInstansiateRate;
        presentTime = 0;
    }

    private void loadSmartBrains()
    {
        while (allBrainsPool.Count < poolSize)
            foreach (Brain brain in SmartBrains.smartBrains)
                allBrainsPool.Add(brain);
        //while (allBrainsPool.Count < poolSize)
        //    allBrainsPool.Add(SmartBrains.smartBrains[0]);
    }

    private void generateRandomPool()
    {
        while (allBrainsPool.Count < poolSize)
        {
            double[,] w1;
            double[,] bias1;
            double[,] w2;
            double[,] bias2;
            double[,] w3;
            double[,] bias3;

            if (gridInput)
            {
                w1 = MatrixUtils.generateRandomMatrix(25, 200, -1, 1);
                bias1 = MatrixUtils.generateRandomMatrix(25, 1, -1, 1);
                w2 = MatrixUtils.generateRandomMatrix(5, 25, -1, 1);
                bias2 = MatrixUtils.generateRandomMatrix(5, 1, -1, 1);
                w3 = MatrixUtils.generateRandomMatrix(5, 5, -1, 1);
                bias3 = MatrixUtils.generateRandomMatrix(5, 1, -1, 1);
            }
            else
            {
                w1 = MatrixUtils.generateRandomMatrix(32, 32, -1, 1);
                bias1 = MatrixUtils.generateRandomMatrix(32, 1, -1, 1);
                w2 = MatrixUtils.generateRandomMatrix(16, 32, -1, 1);
                bias2 = MatrixUtils.generateRandomMatrix(16, 1, -1, 1);
                w3 = MatrixUtils.generateRandomMatrix(5, 16, -1, 1);
                bias3 = MatrixUtils.generateRandomMatrix(5, 1, -1, 1);
            }
            
            Brain brain = new Brain();
            brain.w = w1;
            brain.bias = bias1;
            brain.w2 = w2;
            brain.bias2 = bias2;
            brain.w3 = w3;
            brain.bias3 = bias3;

            allBrainsPool.Add(brain);
        }
    }

    private void startPhaseGame()
    {
        if (gamePhase < gamePhases)
            numberOfGames = 1000;
        else
            numberOfGames = poolSize - 1000 * (gamePhase);

        presentTime = 0;
        InstantiateGames(gamePhase);
        if (!endless)
            StartCoroutine(counter());
    }

    private void InstantiateGames(int gamePhase)
    {
        gameInstances.Clear();
        for (int i = 0; i < numberOfGames; i++)
        {
            gameInstances.Add(GameObject.Instantiate(gameInstance, calculateInstantiatePosition(i, numberOfGames), new Quaternion(), transform));
            Brain brain = ((GameObject)gameInstances[i]).transform.GetChild(0).GetComponent<Brain>();
            brain.w = allBrainsPool[gamePhase * 100 + i].w;
            brain.bias = allBrainsPool[gamePhase * 100 + i].bias;
            brain.w2= allBrainsPool[gamePhase * 100 + i].w2;
            brain.bias2 = allBrainsPool[gamePhase * 100 + i].bias2;
            brain.w3 = allBrainsPool[gamePhase * 100 + i].w3;
            brain.bias3 = allBrainsPool[gamePhase * 100 + i].bias3;
        }
    }

    IEnumerator counter()
    {
        yield return new WaitForSeconds(timeOfPlay);
        CollectPool();
    }

    private void CollectPool()
    {
        addAllPlayedBrainsToAuxPool();
        if (gamePhase < gamePhases)
            gamePhase++;
        else
        {
            presentPosition = enemiesStartPositionInCircle;
            calculateFitness();
            addAprovedBrainsToPool();
            refillBrainsPool();
            printResults();
            gamePhase = 0;
        }
        startPhaseGame();
    }

    private void printResults()
    {
        if (fitneses.Count > 0)
        {
            float max = fitneses[0];
            for (int i = 0; i < poolSize; i++)
                if (max < fitneses[i])
                    max = fitneses[i];

            Debug.Log("Generation:\t" + generation);
            Debug.Log("Fitness Average:\t\t" + (int)fitnessArg);
            Debug.Log("Best:\t\t\t" + (int)max);
            generation++;
        }
    }

    private void refillBrainsPool()
    {
        allBrainsPool.Clear();
        while (allBrainsPool.Count < poolSize)
        {
            int RandPos1 = UnityEngine.Random.Range(0, aprovedBrainsPool.Count);
            int RandPos2 = UnityEngine.Random.Range(0, aprovedBrainsPool.Count);
            Brain Brain = new Brain();
            Brain.w = MatrixUtils.mixRandomlyWithMutation(aprovedBrainsPool[RandPos1].w, aprovedBrainsPool[RandPos2].w, mutationRate, -1, 1);
            Brain.bias = MatrixUtils.mixRandomlyWithMutation(aprovedBrainsPool[RandPos1].bias, aprovedBrainsPool[RandPos2].bias, mutationRate, -1, 1);
            Brain.w2 = MatrixUtils.mixRandomlyWithMutation(aprovedBrainsPool[RandPos1].w2, aprovedBrainsPool[RandPos2].w2, mutationRate, -1, 1);
            Brain.bias2 = MatrixUtils.mixRandomlyWithMutation(aprovedBrainsPool[RandPos1].bias2, aprovedBrainsPool[RandPos2].bias2, mutationRate, -1, 1);
            if (thirLayers)
            {
                Brain.w3 = MatrixUtils.mixRandomlyWithMutation(aprovedBrainsPool[RandPos1].w3, aprovedBrainsPool[RandPos2].w3, mutationRate, -1, 1);
                Brain.bias3 = MatrixUtils.mixRandomlyWithMutation(aprovedBrainsPool[RandPos1].bias3, aprovedBrainsPool[RandPos2].bias3, mutationRate, -1, 1);
            }
            allBrainsPool.Add(Brain);
        }
    }

    private void addAllPlayedBrainsToAuxPool()
    {
        auxBrainsPool.Clear();
        fitneses.Clear();
        foreach (GameObject gameInstance in gameInstances)
        {
            auxBrainsPool.Add(gameInstance.transform.GetChild(0).GetComponent<Brain>());
            fitneses.Add(gameInstance.transform.GetChild(0).GetComponent<PlayerState>().fitness);
            GameObject.Destroy(gameInstance);
        }
        gameInstances.Clear();
    }

    private void addAprovedBrainsToPool()
    {
        aprovedBrainsPool.Clear();
        for (int i = 0; i < poolSize; i++)
        {
            if (fitneses[i] >= fitnessArg-1) {
                for (int j = -1; j < (int) Mathf.Pow(fitneses[i] - fitnessArg, 2); j++)
                    aprovedBrainsPool.Add(auxBrainsPool[i]);
            }
            if (fitneses[i] > 7000)
                HandleRecordBestBrain(auxBrainsPool[i], fitneses[i]);
        }
    }

    private void HandleRecordBestBrain(Brain brain, float fitness)
    {
        DateTime dt = DateTime.Now;
        HandleTextFile.WriteString("\n\ndate: " + dt.ToString("yyyy-MM-dd-hh:mm"));
        HandleTextFile.WriteString("Generation: " + generation);
        HandleTextFile.WriteString("Fitness: " + fitness);
        HandleTextFile.WriteString("Pool Size: " + poolSize);
        if (endless)
            HandleTextFile.WriteString("Time Of Play: Endeless");
        else
            HandleTextFile.WriteString("Time Of Play: " + timeOfPlay);


        HandleTextFile.WriteString(dt.ToString("\nW1"));
        HandleTextFile.WriteString(MatrixUtils.printMatrix(brain.w));
        HandleTextFile.WriteString(dt.ToString("\nBIAS"));
        HandleTextFile.WriteString(MatrixUtils.printMatrix(brain.bias));
        HandleTextFile.WriteString(dt.ToString("\nW2"));
        HandleTextFile.WriteString(MatrixUtils.printMatrix(brain.w2));
        HandleTextFile.WriteString(dt.ToString("\nBIAS2"));
        HandleTextFile.WriteString(MatrixUtils.printMatrix(brain.bias2));
        if (thirLayers){
            HandleTextFile.WriteString(dt.ToString("\nW3"));
            HandleTextFile.WriteString(MatrixUtils.printMatrix(brain.w3));
            HandleTextFile.WriteString(dt.ToString("\nBIAS3"));
            HandleTextFile.WriteString(MatrixUtils.printMatrix(brain.bias3));
        }
    }

    private void calculateFitness()
    {
        fitnessArg = 0;
        for (int i = 0; i < poolSize; i++)
            fitnessArg += fitneses[i];
        fitnessArg = fitnessArg / poolSize;
    }

    void Update()
    {
        if (endless)
        {
            bool atLeastOneAlive = false;
            foreach (GameObject gob in gameInstances)
                atLeastOneAlive = atLeastOneAlive || gob.transform.GetChild(0).GetComponent<PlayerState>().stillAlive;
            if (!atLeastOneAlive && !manual)
                CollectPool();
        }

        presentTime += 0.033f;
        if (presentTime >= timeToSpawn)
        {
            if (random)
                SpawnEnemysRandom();
            else
                SpawnEnemys();
            presentTime = 0;
        }

        float maxFitness = -Mathf.Infinity;
        foreach (GameObject gob in gameInstances)
        {
            float fitness = gob.transform.GetChild(0).GetComponent<PlayerState>().fitness;
            if (maxFitness < fitness && !manual)
            {
                maxFitness = fitness;
                Camera.main.transform.position = new Vector3(gob.transform.position.x, gob.transform.position.y, -1);
            }
        }
    }

    private void SpawnEnemys()
    {
        float x = positions[presentPosition].x;
        float y = positions[presentPosition].y;

        foreach (GameObject gameInstance in gameInstances)
            gameInstance.transform.GetChild(3).GetComponent<EnemySpawner>().SpawnEnemy(x*100, y*100);
        if (manual)
            transform.GetChild(0).GetChild(3).GetComponent<EnemySpawner>().SpawnEnemy(x * 100, y * 100);

        if (presentPosition + 1 < positions.Count)
            presentPosition++;
        else
            presentPosition = 0;
    }

    private void SpawnEnemysRandom()
    {
        float x = 0;
        float y = 0;
        int v = UnityEngine.Random.Range(0, 4);
        if (blockUp && blockDown && blockLeft && blockRight)
        {
            x = UnityEngine.Random.Range(-5, 5);
            y = 5;
        }
        else
            switch (v)
            {
                case 0:
                    if (blockUp)
                    {
                        SpawnEnemysRandom();
                        return;
                    }
                    x = UnityEngine.Random.Range(-5, 5);
                    y = 5;
                    break;
                case 1:
                    if (blockDown)
                    {
                        SpawnEnemysRandom();
                        return;
                    }
                    x = UnityEngine.Random.Range(-5, 5);
                    y = -5;
                    break;
                case 2:
                    if (blockRight)
                    {
                        SpawnEnemysRandom();
                        return;
                    }
                    x = 5;
                    y = UnityEngine.Random.Range(-5, 5);
                    break;
                case 3:
                    if (blockLeft)
                    {
                        SpawnEnemysRandom();
                        return;
                    }
                    x = -5;
                    y = UnityEngine.Random.Range(-5, 5);
                    break;
                default:
                    throw new System.Exception("Wrong arguments");
            }

        foreach (GameObject gameInstance in gameInstances)
            gameInstance.transform.GetChild(3).GetComponent<EnemySpawner>().SpawnEnemy(x*100, y*100);
        if (manual)
            transform.GetChild(0).GetChild(3).GetComponent<EnemySpawner>().SpawnEnemy(x * 100, y * 100);

    }

    private Vector3 calculateInstantiatePosition(int i, int poolSize)
    {
        int hLenght = (int)Mathf.Ceil(Mathf.Sqrt(poolSize));
        float floor = Mathf.Floor(i / hLenght);
        float x = (10 * i - floor * hLenght * 10) + (2 * i - floor * hLenght * 2);
        float y = 10 * floor + (4 * floor);
        return new Vector3(x*100, y*100, 0);
    }

    private void createPostions()
    {
        positions = new List<Vector3>();
        if (circle) {
            //circle
            positions.Add(new Vector3(-5, -5, 0));
            positions.Add(new Vector3(-4, -5, 0));
            positions.Add(new Vector3(-3, -5, 0));
            positions.Add(new Vector3(-2, -5, 0));
            positions.Add(new Vector3(-1, -5, 0));
            positions.Add(new Vector3(0, -5, 0));
            positions.Add(new Vector3(1, -5, 0));
            positions.Add(new Vector3(2, -5, 0));
            positions.Add(new Vector3(3, -5, 0));
            positions.Add(new Vector3(4, -5, 0));
            positions.Add(new Vector3(5, -5, 0));

            positions.Add(new Vector3(5, -4, 0));
            positions.Add(new Vector3(5, -3, 0));
            positions.Add(new Vector3(5, -2, 0));
            positions.Add(new Vector3(5, -1, 0));
            positions.Add(new Vector3(5, 0, 0));
            positions.Add(new Vector3(5, 1, 0));
            positions.Add(new Vector3(5, 2, 0));
            positions.Add(new Vector3(5, 3, 0));
            positions.Add(new Vector3(5, 4, 0));
            positions.Add(new Vector3(5, 5, 0));

            positions.Add(new Vector3(4, 5, 0));
            positions.Add(new Vector3(3, 5, 0));
            positions.Add(new Vector3(2, 5, 0));
            positions.Add(new Vector3(1, 5, 0));
            positions.Add(new Vector3(0, 5, 0));
            positions.Add(new Vector3(-1, 5, 0));
            positions.Add(new Vector3(-2, 5, 0));
            positions.Add(new Vector3(-3, 5, 0));
            positions.Add(new Vector3(-4, 5, 0));
            positions.Add(new Vector3(-5, 5, 0));

            positions.Add(new Vector3(-5, 4, 0));
            positions.Add(new Vector3(-5, 3, 0));
            positions.Add(new Vector3(-5, 2, 0));
            positions.Add(new Vector3(-5, 1, 0));
            positions.Add(new Vector3(-5, 0, 0));
            positions.Add(new Vector3(-5, -1, 0));
            positions.Add(new Vector3(-5, -2, 0));
            positions.Add(new Vector3(-5, -3, 0));
            positions.Add(new Vector3(-5, -4, 0));
            
        } else
        {
            //one by face down-up-left-right
            positions.Add(new Vector3(-5, -5, 0));
            positions.Add(new Vector3(5, 5, 0));
            positions.Add(new Vector3(-5, 5, 0));
            positions.Add(new Vector3(5, -5, 0));
            positions.Add(new Vector3(-4, -5, 0));
            positions.Add(new Vector3(4, 5, 0));
            positions.Add(new Vector3(-5, 4, 0));
            positions.Add(new Vector3(5, -4, 0));
            positions.Add(new Vector3(-3, -5, 0));
            positions.Add(new Vector3(3, 5, 0));
            positions.Add(new Vector3(-5, 3, 0));
            positions.Add(new Vector3(5, -3, 0));
            positions.Add(new Vector3(-2, -5, 0));
            positions.Add(new Vector3(2, 5, 0));
            positions.Add(new Vector3(-5, 2, 0));
            positions.Add(new Vector3(5, -2, 0));
            positions.Add(new Vector3(-1, -5, 0));
            positions.Add(new Vector3(1, 5, 0));
            positions.Add(new Vector3(-5, 1, 0));
            positions.Add(new Vector3(5, -1, 0));
            positions.Add(new Vector3(0, -5, 0));
            positions.Add(new Vector3(0, 5, 0));
            positions.Add(new Vector3(-5, 0, 0));
            positions.Add(new Vector3(5, 0, 0));
            positions.Add(new Vector3(1, -5, 0));
            positions.Add(new Vector3(-1, 5, 0));
            positions.Add(new Vector3(-5, -1, 0));
            positions.Add(new Vector3(5, 1, 0));
            positions.Add(new Vector3(2, -5, 0));
            positions.Add(new Vector3(-2, 5, 0));
            positions.Add(new Vector3(-5, -2, 0));
            positions.Add(new Vector3(5, 2, 0));
            positions.Add(new Vector3(3, -5, 0));
            positions.Add(new Vector3(-3, 5, 0));
            positions.Add(new Vector3(-5, -3, 0));
            positions.Add(new Vector3(5, 3, 0));
            positions.Add(new Vector3(4, -5, 0));
            positions.Add(new Vector3(-4, 5, 0));
            positions.Add(new Vector3(-5, -4, 0));
            positions.Add(new Vector3(5, 4, 0));
            //circle:
            positions.Add(new Vector3(-5, -5, 0));
            positions.Add(new Vector3(-4, -5, 0));
            positions.Add(new Vector3(-3, -5, 0));
            positions.Add(new Vector3(-2, -5, 0));
            positions.Add(new Vector3(-1, -5, 0));
            positions.Add(new Vector3(0, -5, 0));
            positions.Add(new Vector3(1, -5, 0));
            positions.Add(new Vector3(2, -5, 0));
            positions.Add(new Vector3(3, -5, 0));
            positions.Add(new Vector3(4, -5, 0));
            positions.Add(new Vector3(5, -5, 0));

            positions.Add(new Vector3(5, -4, 0));
            positions.Add(new Vector3(5, -3, 0));
            positions.Add(new Vector3(5, -2, 0));
            positions.Add(new Vector3(5, -1, 0));
            positions.Add(new Vector3(5, 0, 0));
            positions.Add(new Vector3(5, 1, 0));
            positions.Add(new Vector3(5, 2, 0));
            positions.Add(new Vector3(5, 3, 0));
            positions.Add(new Vector3(5, 4, 0));
            positions.Add(new Vector3(5, 5, 0));

            positions.Add(new Vector3(4, 5, 0));
            positions.Add(new Vector3(3, 5, 0));
            positions.Add(new Vector3(2, 5, 0));
            positions.Add(new Vector3(1, 5, 0));
            positions.Add(new Vector3(0, 5, 0));
            positions.Add(new Vector3(-1, 5, 0));
            positions.Add(new Vector3(-2, 5, 0));
            positions.Add(new Vector3(-3, 5, 0));
            positions.Add(new Vector3(-4, 5, 0));
            positions.Add(new Vector3(-5, 5, 0));

            positions.Add(new Vector3(-5, 4, 0));
            positions.Add(new Vector3(-5, 3, 0));
            positions.Add(new Vector3(-5, 2, 0));
            positions.Add(new Vector3(-5, 1, 0));
            positions.Add(new Vector3(-5, 0, 0));
            positions.Add(new Vector3(-5, -1, 0));
            positions.Add(new Vector3(-5, -2, 0));
            positions.Add(new Vector3(-5, -3, 0));
            positions.Add(new Vector3(-5, -4, 0));
            //one by face: up-right-down-left
            //positions.Add(new Vector3(5, -5, 0));
            //positions.Add(new Vector3(-5, 5, 0));
            //positions.Add(new Vector3(-5, -5, 0));
            //positions.Add(new Vector3(5, 5, 0));
            //positions.Add(new Vector3(5, -4, 0));
            //positions.Add(new Vector3(-5, 4, 0));
            //positions.Add(new Vector3(-4, -5, 0));
            //positions.Add(new Vector3(4, 5, 0));
            //positions.Add(new Vector3(5, -3, 0));
            //positions.Add(new Vector3(-5, 3, 0));
            //positions.Add(new Vector3(-3, -5, 0));
            //positions.Add(new Vector3(3, 5, 0));
            //positions.Add(new Vector3(5, -2, 0));
            //positions.Add(new Vector3(-5, 2, 0));
            //positions.Add(new Vector3(-2, -5, 0));
            //positions.Add(new Vector3(2, 5, 0));
            //positions.Add(new Vector3(5, -1, 0));
            //positions.Add(new Vector3(-5, 1, 0));
            //positions.Add(new Vector3(-1, -5, 0));
            //positions.Add(new Vector3(1, 5, 0));
            //positions.Add(new Vector3(5, 0, 0));
            //positions.Add(new Vector3(-5, 0, 0));
            //positions.Add(new Vector3(0, -5, 0));
            //positions.Add(new Vector3(0, 5, 0));
            //positions.Add(new Vector3(5, 1, 0));
            //positions.Add(new Vector3(-5, -1, 0));
            //positions.Add(new Vector3(1, -5, 0));
            //positions.Add(new Vector3(-1, 5, 0));
            //positions.Add(new Vector3(5, 2, 0));
            //positions.Add(new Vector3(-5, -2, 0));
            //positions.Add(new Vector3(2, -5, 0));
            //positions.Add(new Vector3(-2, 5, 0));
            //positions.Add(new Vector3(5, 3, 0));
            //positions.Add(new Vector3(-5, -3, 0));
            //positions.Add(new Vector3(3, -5, 0));
            //positions.Add(new Vector3(-3, 5, 0));
            //positions.Add(new Vector3(5, 4, 0));
            //positions.Add(new Vector3(-5, -4, 0));
            //positions.Add(new Vector3(4, -5, 0));
            //positions.Add(new Vector3(-4, 5, 0));
            ////one by face down-left-right-up
            //positions.Add(new Vector3(-5, -5, 0));
            //positions.Add(new Vector3(5, 5, 0));
            //positions.Add(new Vector3(-5, 5, 0));
            //positions.Add(new Vector3(5, -5, 0));
            //positions.Add(new Vector3(-4, -5, 0));
            //positions.Add(new Vector3(4, 5, 0));
            //positions.Add(new Vector3(-5, 4, 0));
            //positions.Add(new Vector3(5, -4, 0));
            //positions.Add(new Vector3(-3, -5, 0));
            //positions.Add(new Vector3(3, 5, 0));
            //positions.Add(new Vector3(-5, 3, 0));
            //positions.Add(new Vector3(5, -3, 0));
            //positions.Add(new Vector3(-2, -5, 0));
            //positions.Add(new Vector3(2, 5, 0));
            //positions.Add(new Vector3(-5, 2, 0));
            //positions.Add(new Vector3(5, -2, 0));
            //positions.Add(new Vector3(-1, -5, 0));
            //positions.Add(new Vector3(1, 5, 0));
            //positions.Add(new Vector3(-5, 1, 0));
            //positions.Add(new Vector3(5, -1, 0));
            //positions.Add(new Vector3(0, -5, 0));
            //positions.Add(new Vector3(0, 5, 0));
            //positions.Add(new Vector3(-5, 0, 0));
            //positions.Add(new Vector3(5, 0, 0));
            //positions.Add(new Vector3(1, -5, 0));
            //positions.Add(new Vector3(-1, 5, 0));
            //positions.Add(new Vector3(-5, -1, 0));
            //positions.Add(new Vector3(5, 1, 0));
            //positions.Add(new Vector3(2, -5, 0));
            //positions.Add(new Vector3(-2, 5, 0));
            //positions.Add(new Vector3(-5, -2, 0));
            //positions.Add(new Vector3(5, 2, 0));
            //positions.Add(new Vector3(3, -5, 0));
            //positions.Add(new Vector3(-3, 5, 0));
            //positions.Add(new Vector3(-5, -3, 0));
            //positions.Add(new Vector3(5, 3, 0));
            //positions.Add(new Vector3(4, -5, 0));
            //positions.Add(new Vector3(-4, 5, 0));
            //positions.Add(new Vector3(-5, -4, 0));
            //positions.Add(new Vector3(5, 4, 0));
            ////two by face down-left-right-up
            //positions.Add(new Vector3(-5, -5, 0));
            //positions.Add(new Vector3(0, -5, 0));
            //positions.Add(new Vector3(5, -5, 0));
            //positions.Add(new Vector3(5, 0, 0));
            //positions.Add(new Vector3(5, 5, 0));
            //positions.Add(new Vector3(0, 5, 0));
            //positions.Add(new Vector3(-5, 5, 0));
            //positions.Add(new Vector3(-5, 0, 0));
            //positions.Add(new Vector3(-4, -5, 0));
            //positions.Add(new Vector3(1, -5, 0));
            //positions.Add(new Vector3(5, -4, 0));
            //positions.Add(new Vector3(5, 1, 0));
            //positions.Add(new Vector3(4, 5, 0));
            //positions.Add(new Vector3(-1, 5, 0));
            //positions.Add(new Vector3(-5, 4, 0));
            //positions.Add(new Vector3(-5, -1, 0));
            //positions.Add(new Vector3(-3, -5, 0));
            //positions.Add(new Vector3(2, -5, 0));
            //positions.Add(new Vector3(5, -3, 0));
            //positions.Add(new Vector3(5, 2, 0));
            //positions.Add(new Vector3(3, 5, 0));
            //positions.Add(new Vector3(-2, 5, 0));
            //positions.Add(new Vector3(-2, -5, 0));
            //positions.Add(new Vector3(3, -5, 0));
            //positions.Add(new Vector3(5, -2, 0));
            //positions.Add(new Vector3(5, 3, 0));
            //positions.Add(new Vector3(-5, 3, 0));
            //positions.Add(new Vector3(-5, -2, 0));
            //positions.Add(new Vector3(2, 5, 0));
            //positions.Add(new Vector3(-3, 5, 0));
            //positions.Add(new Vector3(-5, 2, 0));
            //positions.Add(new Vector3(-5, -3, 0));
            //positions.Add(new Vector3(-1, -5, 0));
            //positions.Add(new Vector3(4, -5, 0));
            //positions.Add(new Vector3(5, -1, 0));
            //positions.Add(new Vector3(5, 4, 0));
            //positions.Add(new Vector3(1, 5, 0));
            //positions.Add(new Vector3(-4, 5, 0));
            //positions.Add(new Vector3(-5, 1, 0));
            //positions.Add(new Vector3(-5, -4, 0));
            ////random
            //positions.Add(new Vector3(-5, -5, 0));
            //positions.Add(new Vector3(5, 2, 0));
            //positions.Add(new Vector3(5, 3, 0));
            //positions.Add(new Vector3(4, 5, 0));
            //positions.Add(new Vector3(3, 5, 0));
            //positions.Add(new Vector3(-4, -5, 0));
            //positions.Add(new Vector3(-5, 1, 0));
            //positions.Add(new Vector3(-5, 0, 0));
            //positions.Add(new Vector3(-5, 3, 0));
            //positions.Add(new Vector3(-5, 2, 0));
            //positions.Add(new Vector3(-3, -5, 0));
            //positions.Add(new Vector3(1, 5, 0));
            //positions.Add(new Vector3(0, 5, 0));
            //positions.Add(new Vector3(-5, -1, 0));
            //positions.Add(new Vector3(-5, -4, 0));
            //positions.Add(new Vector3(-2, -5, 0));
            //positions.Add(new Vector3(-1, -5, 0));
            //positions.Add(new Vector3(-4, 5, 0));
            //positions.Add(new Vector3(-5, 5, 0));
            //positions.Add(new Vector3(2, 5, 0));
            //positions.Add(new Vector3(0, -5, 0));
            //positions.Add(new Vector3(1, -5, 0));
            //positions.Add(new Vector3(-1, 5, 0));
            //positions.Add(new Vector3(-2, 5, 0));
            //positions.Add(new Vector3(5, 0, 0));
            //positions.Add(new Vector3(5, 1, 0));
            //positions.Add(new Vector3(2, -5, 0));
            //positions.Add(new Vector3(-5, -2, 0));
            //positions.Add(new Vector3(-5, -3, 0));
            //positions.Add(new Vector3(5, -2, 0));
            //positions.Add(new Vector3(5, -1, 0));
            //positions.Add(new Vector3(3, -5, 0));
            //positions.Add(new Vector3(-5, 4, 0));
            //positions.Add(new Vector3(4, -5, 0));
            //positions.Add(new Vector3(5, 4, 0));
            //positions.Add(new Vector3(5, 5, 0));
            //positions.Add(new Vector3(5, -5, 0));
            //positions.Add(new Vector3(5, -4, 0));
            //positions.Add(new Vector3(5, -3, 0));
            //positions.Add(new Vector3(-3, 5, 0));
        }
    }

}
