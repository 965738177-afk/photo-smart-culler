package com.miaoxuan.mobile.data

import android.content.Context
import com.miaoxuan.mobile.model.TaskSummary
import org.json.JSONArray
import org.json.JSONObject

/** Stores only compact task summaries. Image URIs and pixels are never copied into history. */
class TaskHistoryStore(context: Context) {
    private val preferences = context.getSharedPreferences("miaoxuan_history", Context.MODE_PRIVATE)

    fun load(): List<TaskSummary> = runCatching {
        val stored = JSONArray(preferences.getString(KEY, "[]"))
        buildList {
            for (index in 0 until stored.length()) {
                val item = stored.getJSONObject(index)
                add(
                    TaskSummary(
                        title = item.getString("title"),
                        analyzedAtMillis = item.getLong("analyzedAtMillis"),
                        sourceCount = item.getInt("sourceCount"),
                        keepCount = item.getInt("keepCount"),
                    ),
                )
            }
        }
    }.getOrDefault(emptyList())

    fun append(summary: TaskSummary) {
        val all = (listOf(summary) + load()).take(MAX_ENTRIES)
        val encoded = JSONArray().apply {
            all.forEach { entry ->
                put(JSONObject().apply {
                    put("title", entry.title)
                    put("analyzedAtMillis", entry.analyzedAtMillis)
                    put("sourceCount", entry.sourceCount)
                    put("keepCount", entry.keepCount)
                })
            }
        }
        preferences.edit().putString(KEY, encoded.toString()).apply()
    }

    companion object {
        private const val KEY = "task_history"
        private const val MAX_ENTRIES = 20
    }
}

