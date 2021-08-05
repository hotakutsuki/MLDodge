using UnityEngine;
using UnityEditor;
using System.IO;

public class HandleTextFile : MonoBehaviour
{
    //public TextAsset asset;
    [MenuItem("Tools/Write file")]

    public static void WriteString(string s)
    {
        string path = "Assets/Brains/brain.txt";

        StreamWriter writer = new StreamWriter(path, true);
        writer.WriteLine(s);
        writer.Close();

        ////Re-import the file to update the reference in the editor
        //AssetDatabase.ImportAsset(path);
        //TextAsset asset = (TextAsset) Resources.Load("test");

        ////Print the text from the file
        //Debug.Log(asset.text);
    }

    [MenuItem("Tools/Read file")]
    public static string ReadString()
    {
        string path = "Assets/Brains/brain.txt";

        //Read the text from directly from the test.txt file
        StreamReader reader = new StreamReader(path);
        string s = reader.ReadToEnd();
        Debug.Log(s);
        reader.Close();
        return s;
    }

}