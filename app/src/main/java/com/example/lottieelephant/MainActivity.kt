package com.example.lottieelephant

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieClipSpec
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.rememberLottieAnimatable
import com.airbnb.lottie.compose.rememberLottieComposition

/** Marker names defined in res/raw/elephant.json (see tools/generate_elephant.py). */
private const val MARKER_JUMP = "jump"
private const val MARKER_IDLE = "idle"

/** Width / height of the Lottie composition. */
private const val ELEPHANT_ASPECT_RATIO = 580f / 500f

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme {
                ElephantScreen()
            }
        }
    }
}

@Composable
fun ElephantScreen() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    0f to Color(0xFF8EC9FF),
                    0.75f to Color(0xFFD0E8FF),
                    0.75f to Color(0xFF7CC36A),
                    1f to Color(0xFF5FA552),
                )
            )
            .safeDrawingPadding(),
    ) {
        Text(
            text = stringResource(R.string.tap_hint),
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 48.dp),
            color = Color(0xFF2B3A55),
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
        )
        JumpingElephant(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                // the ground line of the animation (y = 430 of 500) sits on the grass edge
                .padding(bottom = 24.dp)
                .fillMaxWidth()
                .aspectRatio(ELEPHANT_ASPECT_RATIO),
        )
    }
}

/**
 * The elephant breathes and flaps its ears in an endless idle loop.
 * A tap plays the "jump" segment once (crouch -> push-off -> flight -> landing) and then
 * returns to the idle loop. Taps during a jump are ignored.
 */
@Composable
fun JumpingElephant(modifier: Modifier = Modifier) {
    val composition by rememberLottieComposition(LottieCompositionSpec.RawRes(R.raw.elephant))
    val animatable = rememberLottieAnimatable()
    var jumpRequests by remember { mutableIntStateOf(0) }
    var isJumping by remember { mutableStateOf(false) }

    LaunchedEffect(composition, jumpRequests) {
        val comp = composition ?: return@LaunchedEffect
        if (jumpRequests > 0) {
            isJumping = true
            try {
                animatable.animate(
                    composition = comp,
                    clipSpec = LottieClipSpec.Marker(MARKER_JUMP),
                    initialProgress = 0f,
                    iterations = 1,
                )
            } finally {
                isJumping = false
            }
        }
        animatable.animate(
            composition = comp,
            clipSpec = LottieClipSpec.Marker(MARKER_IDLE),
            iterations = LottieConstants.IterateForever,
        )
    }

    LottieAnimation(
        composition = composition,
        progress = { animatable.progress },
        modifier = modifier.clickable(
            interactionSource = remember { MutableInteractionSource() },
            indication = null,
            onClickLabel = stringResource(R.string.elephant_description),
            role = Role.Button,
        ) {
            if (!isJumping) jumpRequests++
        },
    )
}

@Preview(showBackground = true)
@Composable
private fun ElephantScreenPreview() {
    MaterialTheme {
        ElephantScreen()
    }
}
